from flask import Flask, request, jsonify, send_file, render_template, Response, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import uuid
import csv
from io import BytesIO, StringIO
import numpy as np
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import json
import logging
import queue
import re
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Enable CORS
from flask_cors import CORS
CORS(app)

# Configure Gemini API
API_KEY = 'AIzaSyBFBenEhzVsRRSeG0xfCfibz-Jn2EsJjNI'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')
global from_info

# Global variables for execution control
stop_execution_flag = False
stop_execution_lock = threading.Lock()
active_threads = {}  # Track active processing threads
threads_lock = threading.Lock()  # Lock for thread dictionary
latest_products = []  # Store the latest processed products
product_data_store = {}  # Store product data in memory instead of files

class SSEManager:
    def __init__(self):
        self.clients = {}
        self.lock = threading.Lock()
    
    def add_client(self, task_id):
        with self.lock:
            if task_id not in self.clients:
                self.clients[task_id] = queue.Queue()
            return self.clients[task_id]
    
    def remove_client(self, task_id):
        with self.lock:
            if task_id in self.clients:
                del self.clients[task_id]
    
    def send_event(self, task_id, event_type, data):
        with self.lock:
            if task_id in self.clients:
                event = {
                    'id': str(uuid.uuid4()),
                    'event': event_type,
                    'data': json.dumps(data)
                }
                self.clients[task_id].put(event)

sse_manager = SSEManager()
progress_data = {}

class StrictRateLimiter:
    def __init__(self, rate_per_minute=60):
        self.rate = rate_per_minute
        self.interval = 60.0 / rate_per_minute
        self.last_request_time = 0
        self.lock = threading.Lock()
        self.request_count = 0
        self.reset_time = time.time() + 60
        self.emergency_buffer = 5  # Keep 5 requests as buffer
        
    def wait(self):
        with self.lock:
            current_time = time.time()
            
            # Reset counter if minute has passed
            if current_time > self.reset_time:
                self.request_count = 0
                self.reset_time = current_time + 60
            
            # If we're approaching the limit, wait until next reset
            if self.request_count >= (self.rate - self.emergency_buffer):
                sleep_time = self.reset_time - current_time
                if sleep_time > 0:
                    logger.warning(f"Approaching rate limit. Sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                self.request_count = 0
                self.reset_time = time.time() + 60
            
            # Enforce interval between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.interval:
                sleep_time = self.interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = current_time
            self.request_count += 1
            return current_time

gemini_limiter = StrictRateLimiter(rate_per_minute=60)

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_server_error(e):
    return redirect(url_for('index'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_value(value):
    if pd.isna(value) or value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return str(value)

def simple_shorten(description, max_words=8):
    words = re.sub(r'[^\w\s]', '', description).split()[:max_words]
    return ' '.join(words)

def extract_from_info(file_content, filename):
    """Extract from information from the provided file content."""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        else:
            df = pd.read_excel(BytesIO(file_content))
        
        if len(df) > 0:
            first_row = df.iloc[0]
            global from_info
            from_info = {
                'FromName': safe_value(first_row.get('FromName', '')),
                'FromCompany': safe_value(first_row.get('FromCompany', '')),
                'FromStreet': safe_value(first_row.get('FromStreet', '')),
                'FromStreet2': safe_value(first_row.get('FromStreet2', '')),
                'FromCity': safe_value(first_row.get('FromCity', '')),
                'FromState': safe_value(first_row.get('FromState', '')),
                'FromZip': safe_value(first_row.get('FromZip', '')),
                'FromPhone': safe_value(first_row.get('FromPhone', ''))
            }
            logger.info(f"Successfully extracted 'from' information: {from_info['FromName']}, {from_info['FromCity']}")
            return True
    except Exception as e:
        logger.error(f"Error extracting 'from' information: {str(e)}")
        return False

def shorten_description(description, task_id, is_retry=False):
    global stop_execution_flag
    
    if stop_execution_flag:
        with stop_execution_lock:
            if stop_execution_flag:
                raise Exception("Processing stopped by user")
    
    if len(description) <= 30:
        return description
        
    try:
        if not is_retry:
            if stop_execution_flag:
                with stop_execution_lock:
                    if stop_execution_flag:
                        raise Exception("Processing stopped by user")
            
            gemini_limiter.wait()
        
        if stop_execution_flag:
            with stop_execution_lock:
                if stop_execution_flag:
                    raise Exception("Processing stopped by user")

        prompt = (
            "You are a product description summarizer. "
            "Given the full product description below, generate a **single concise summary** "
            "in 5 to 10 words that captures all key product identifiers, such as brand, scent, size, and type. "
            "Do NOT provide multiple options, lists, or explanations. "
            "Do NOT start with phrases like 'Here are a few options'. "
            "Do NOT use bullet points or numbering. "
            "Return only the concise summary, nothing else.\n\n"
            f"Product description: {description}"
        )

        response = model.generate_content(prompt)
        shortened = response.text.strip('"\'')
        
        if not shortened or len(shortened) < 3:
            raise Exception("Empty response from AI")
            
        with progress_data[task_id]['ai_lock']:
            progress_data[task_id]['ai_success'] += 1
            
        return shortened
    except Exception as e:
        with stop_execution_lock:
            if not stop_execution_flag:
                logger.error(f"Error shortening description: {str(e)}")
        return None

def generate_product_csv(product_data):
    headers = [
        'FromName', 'FromCompany', 'FromStreet', 'FromStreet2', 'FromCity', 'FromState', 
        'FromZip', 'FromPhone', 'ToName', 'ToCompany', 'ToStreet', 'ToStreet2', 
        'ToCity', 'ToState', 'ToZip', 'ToPhone', 'Weight', 'Length', 'Width', 
        'Height', 'Description', 'order num', 'Reference2', 'Signature'
    ]
    
    row = [
        from_info['FromName'], from_info['FromCompany'], from_info['FromStreet'], 
        from_info['FromStreet2'], from_info['FromCity'], from_info['FromState'], 
        from_info['FromZip'], from_info['FromPhone'],
        product_data['to_info']['ToName'],
        product_data['to_info']['ToCompany'],
        product_data['to_info']['ToStreet'],
        product_data['to_info']['ToStreet2'],
        product_data['to_info']['ToCity'],
        product_data['to_info']['ToState'],
        product_data['to_info']['ToZip'],
        product_data['to_info']['ToPhone'],
        product_data['weight'],
        product_data['length'],
        product_data['width'],
        product_data['height'],
        product_data['short_description'],
        product_data['order_number'],
        product_data['po_number'],
        ''
    ]
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerow(row)
    
    return output.getvalue().encode('utf-8')

def generate_shorter_filename(product_data):
    """Generate a shorter, more concise filename for the CSV file."""
    po_num = product_data['po_number'][-5:] if len(product_data['po_number']) > 5 else product_data['po_number']
    order_num = product_data['order_number'][-5:] if len(product_data['order_number']) > 5 else product_data['order_number']
    
    if 'short_description' in product_data:
        desc = product_data['short_description']
        desc = re.sub(r'\s*\(Qty:.*?\)', '', desc)
        desc = re.sub(r'\s*\(See Details\)', '', desc)
    else:
        desc = product_data.get('description', '')
    
    words = re.findall(r'\w+', desc)
    short_desc = ' '.join(words[:3])
    customer_name = product_data['to_info']['ToName'].split()[0] if product_data['to_info']['ToName'] else 'unknown'
    
    clean_text = re.sub(r'[^\w\s]', '', f"{short_desc}_{customer_name}")
    clean_text = re.sub(r'\s+', '_', clean_text)
    short_uuid = uuid.uuid4().hex[:6]
    
    filename = f"prod_{clean_text}_{short_uuid}.csv"
    return filename

def process_product(product_data, task_id, is_retry=False):
    try:
        if not is_retry:
            with progress_data[task_id]['lock']:
                progress_data[task_id]['current'] += 1
                current = progress_data[task_id]['current']
                total = progress_data[task_id]['total']
                progress_data[task_id]['status'] = f"Processing {current}/{total}"
        
        if len(product_data.get('description', '')) > 0:
            short_desc = shorten_description(product_data['description'], task_id, is_retry)
            
            if short_desc is None:
                return None, product_data
        else:
            short_desc = "Unknown Product"
        
        product_data['short_description'] = f"{short_desc} (Qty: {product_data['qty']})"
        csv_content = generate_product_csv(product_data)
        
        filename = generate_shorter_filename(product_data)
        
        # Store in memory instead of writing to file
        product_data_store[filename] = csv_content
        
        return {
            'product': product_data['short_description'],
            'filename': filename,
            'download_url': f'/download/{filename}',
            'to_info': product_data['to_info']
        }, None
    
    except Exception as e:
        logger.error(f"Error processing product: {str(e)}")
        return None, product_data

def process_po_file(file_content, filename, task_id):
    try:
        if filename.endswith('.csv'):
            chunksize = 1000
            df = pd.concat([chunk for chunk in pd.read_csv(StringIO(file_content.decode('utf-8')), chunksize=chunksize)])
        else:
            df = pd.read_excel(BytesIO(file_content))
        
        original_data = df.to_dict('records')
        records = df.to_dict('records')
        results = []
        
        for row in records:
            try:
                results.append({
                    'product': safe_value(row.get('Item Description', '')),
                    'to_info': {
                        'ToName': safe_value(row.get('Customer Name', '')),
                        'ToCompany': safe_value(row.get('Customer Company', '')),
                        'ToStreet': safe_value(row.get('Ship to Address 1', '')),
                        'ToStreet2': safe_value(row.get('Ship to Address 2', '')),
                        'ToCity': safe_value(row.get('City', '')),
                        'ToState': safe_value(row.get('State', '')),
                        'ToZip': safe_value(row.get('Zip', '')),
                        'ToPhone': safe_value(row.get('Customer Phone Number', ''))
                    },
                    'description': safe_value(row.get('Item Description', '')),
                    'weight': safe_value(row.get('Weight', '')),
                    'length': safe_value(row.get('Length', '')),
                    'width': safe_value(row.get('Width', '')),
                    'height': safe_value(row.get('Height', '')),
                    'po_number': safe_value(row.get('PO#', '')),
                    'order_number': safe_value(row.get('Order#', '')),
                    'qty': safe_value(row.get('Qty', 1))
                })
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue
        
        with progress_data[task_id]['lock']:
            progress_data[task_id]['po_data'] = original_data

        return results
        
    except Exception as e:
        logger.error(f"Error processing PO file: {str(e)}")
        return {'error': str(e)}

def send_progress_update(task_id):
    if task_id in progress_data:
        with progress_data[task_id]['lock']:
            data_copy = {
                'current': progress_data[task_id]['current'],
                'total': progress_data[task_id]['total'],
                'status': progress_data[task_id]['status'],
                'products': progress_data[task_id]['products'].copy(),
                'ai_success': progress_data[task_id]['ai_success'],
                'ai_failed': progress_data[task_id]['ai_failed'],
                'ai_pending': len(progress_data[task_id]['ai_retry'])
            }
            
            if progress_data[task_id].get('success'):
                data_copy['success'] = True
            if progress_data[task_id].get('error'):
                data_copy['error'] = progress_data[task_id]['error']
        
        sse_manager.send_event(task_id, 'progress', {'progress': data_copy})

def background_task(file_content, filename, task_id):
    global stop_execution_flag, latest_products
    
    with threads_lock:
        active_threads[task_id] = {
            'thread': threading.current_thread(),
            'start_time': time.time()
        }
    
    try:
        with stop_execution_lock:
            if stop_execution_flag:
                with progress_data[task_id]['lock']:
                    progress_data[task_id]['error'] = 'Processing stopped before starting'
                    progress_data[task_id]['status'] = 'Cancelled'
                send_progress_update(task_id)
                return
            
        results = process_po_file(file_content, filename, task_id)
        if isinstance(results, dict) and 'error' in results:
            with progress_data[task_id]['lock']:
                progress_data[task_id]['error'] = results['error']
            send_progress_update(task_id)
            return
        
        with progress_data[task_id]['lock']:
            progress_data[task_id]['total'] = len(results)
            progress_data[task_id]['ai_retry'] = []
        send_progress_update(task_id)
        
        max_workers = 4
        processed_count = 0
        retry_items = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_product, product_data, task_id): product_data for product_data in results}
            
            for future in as_completed(futures):
                product_result, retry_data = future.result()
                
                if product_result:
                    with progress_data[task_id]['lock']:
                        progress_data[task_id]['products'].append(product_result)
                    processed_count += 1
                    send_progress_update(task_id)
                
                if retry_data:
                    retry_items.append(retry_data)
            
            max_retries = 100
            retry_count = 0
            
            while retry_items and retry_count < max_retries:
                retry_count += 1
                logger.info(f"Retry attempt {retry_count} with {len(retry_items)} items")
                
                successful_retries = []
                new_retries = []
                
                for retry_data in retry_items:
                    product_result, new_retry_data = process_product(retry_data, task_id, is_retry=True)
                    
                    if product_result:
                        with progress_data[task_id]['lock']:
                            progress_data[task_id]['products'].append(product_result)
                        successful_retries.append(retry_data)
                        processed_count += 1
                        send_progress_update(task_id)
                    elif new_retry_data:
                        new_retries.append(new_retry_data)
                    
                    time.sleep(0.5)
                
                retry_items = new_retries
                
                if not retry_items:
                    break
                
                if retry_items and retry_count < max_retries:
                    wait_time = min(10, 2 ** retry_count)
                    logger.info(f"Waiting {wait_time} seconds before next retry")
                    time.sleep(wait_time)
            
            if retry_items:
                logger.warning(f"Falling back to simple shortening for {len(retry_items)} items")
                
                for retry_data in retry_items:
                    try:
                        short_desc = simple_shorten(retry_data['description'])
                        retry_data['short_description'] = f"{short_desc} (Qty: {retry_data['qty']})"
                        csv_content = generate_product_csv(retry_data)
                        
                        clean_desc = ''.join(c if c.isalnum() else '_' for c in retry_data['short_description'])[:40]
                        filename = f"product_{retry_data['po_number']}_{retry_data['order_number']}_{clean_desc}_{uuid.uuid4().hex[:8]}.csv"
                        
                        # Store in memory instead of writing to file
                        product_data_store[filename] = csv_content
                        
                        with progress_data[task_id]['lock']:
                            progress_data[task_id]['products'].append({
                                'product': retry_data['short_description'],
                                'filename': filename,
                                'download_url': f'/download/{filename}',
                                'to_info': retry_data['to_info']
                            })
                            progress_data[task_id]['ai_failed'] += 1
                        
                        send_progress_update(task_id)
                    except Exception as e:
                        logger.error(f"Error processing fallback product: {str(e)}")
        
        with progress_data[task_id]['lock']:
            if processed_count == progress_data[task_id]['total']:
                progress_data[task_id]['status'] = 'Completed'
                progress_data[task_id]['success'] = True
            else:
                progress_data[task_id]['status'] = 'Failed - Not all items processed'
                progress_data[task_id]['error'] = 'Could not process all items'
        
        send_progress_update(task_id)
        latest_products = results
        
    except Exception as e:
        with stop_execution_lock:
            if stop_execution_flag:
                with progress_data[task_id]['lock']:
                    progress_data[task_id]['error'] = 'Processing stopped by user'
                    progress_data[task_id]['status'] = 'Cancelled'
            else:
                with progress_data[task_id]['lock']:
                    progress_data[task_id]['error'] = str(e)
        send_progress_update(task_id)
        logger.error(f"Error in background task: {str(e)}")
    finally:
        with threads_lock:
            if task_id in active_threads:
                del active_threads[task_id]

def generate_group_csv(product_group, headers):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for product_data in product_group:
        row = [
            from_info['FromName'], from_info['FromCompany'], from_info['FromStreet'],
            from_info['FromStreet2'], from_info['FromCity'], from_info['FromState'],
            from_info['FromZip'], from_info['FromPhone'],
            product_data['to_info']['ToName'],
            product_data['to_info']['ToCompany'],
            product_data['to_info']['ToStreet'],
            product_data['to_info']['ToStreet2'],
            product_data['to_info']['ToCity'],
            product_data['to_info']['ToState'],
            product_data['to_info']['ToZip'],
            product_data['to_info']['ToPhone'],
            product_data['weight'],
            product_data['length'],
            product_data['width'],
            product_data['height'],
            product_data['short_description'],
            product_data['order_number'],
            product_data['po_number'],
            ''
        ]
        writer.writerow(row)
    return output.getvalue().encode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    task_id = request.args.get('channel')
    if not task_id or task_id not in progress_data:
        return "Task not found", 404
    
    def generate():
        q = sse_manager.add_client(task_id)
        
        yield "Content-Type: text/event-stream\n"
        yield "Cache-Control: no-cache\n"
        yield "Connection: keep-alive\n\n"
        
        send_progress_update(task_id)
        
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    yield f"id: {event['id']}\n"
                    yield f"event: {event['event']}\n"
                    yield f"data: {event['data']}\n\n"
                    
                    data = json.loads(event['data'])
                    if 'progress' in data and (data['progress'].get('success') or data['progress'].get('error')):
                        break
                        
                except queue.Empty:
                    yield ": ping\n\n"
                
            sse_manager.remove_client(task_id)
            
        except GeneratorExit:
            sse_manager.remove_client(task_id)
        
    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop_execution', methods=['POST'])
def stop_execution():
    """Endpoint to stop all Gemini API processing"""
    global stop_execution_flag
    
    with stop_execution_lock:
        stop_execution_flag = True
        
    with threads_lock:
        for task_id, thread_info in list(active_threads.items()):
            if thread_info['thread'].is_alive():
                with progress_data[task_id]['lock']:
                    progress_data[task_id]['error'] = 'Processing stopped by user'
                    progress_data[task_id]['status'] = 'Cancelled'
                
                with sse_manager.lock:
                    sse_manager.send_event(task_id, 'stop', {'message': 'Execution stopped by user'})
    
    return jsonify({'success': True, 'message': 'Stopping all execution'})

@app.route('/reset_execution_flag', methods=['POST'])
def reset_execution_flag():
    global stop_execution_flag
    with stop_execution_lock:
        stop_execution_flag = False
    # Clean up finished threads
    with threads_lock:
        for task_id in list(active_threads.keys()):
            if not active_threads[task_id]['thread'].is_alive():
                del active_threads[task_id]
    return jsonify({'success': True, 'message': 'Execution flag reset'})


def update_gemini_api_key(new_key):
    global current_api_key, model
    current_api_key = new_key
    genai.configure(api_key=new_key)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Re-create with new config

@app.route('/update_api_key', methods=['POST'])
def update_api_key():
    try:
        data = request.get_json()
        new_key = data.get('api_key', '').strip()
        if not new_key:
            return jsonify({'error': 'API key cannot be empty'}), 400
        update_gemini_api_key(new_key)
        return jsonify({'success': True, 'message': 'API key updated successfully'})
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    global product_data_store
    product_data_store = {}  # Clear previous data

    user_api_key = request.form.get('gemini_api_key', '').strip()
    if user_api_key:
        try:
            genai.configure(api_key=user_api_key)
            logger.info("Using user-provided API key")
        except Exception as e:
            logger.error(f"Error configuring with user API key: {str(e)}")
            return jsonify({'error': 'Invalid API key provided'}), 400
    
    if 'po_file' not in request.files:
        return jsonify({'error': 'No PO file provided'}), 400
    
    po_file = request.files['po_file']
    
    if po_file.filename == '':
        return jsonify({'error': 'No PO file selected'}), 400
    
    if not allowed_file(po_file.filename):
        return jsonify({'error': 'PO file type not allowed'}), 400
    
    if 'from_file' not in request.files:
        return jsonify({'error': 'No From information file provided'}), 400
    
    from_file = request.files['from_file']
    
    if from_file.filename == '':
        return jsonify({'error': 'No From information file selected'}), 400
    
    if not allowed_file(from_file.filename):
        return jsonify({'error': 'From file type not allowed'}), 400
    
    # Read file contents into memory
    from_file_content = from_file.read()
    if not extract_from_info(from_file_content, from_file.filename):
        return jsonify({'error': 'Failed to extract from information'}), 400
    
    po_file_content = po_file.read()
    
    task_id = str(uuid.uuid4())
    progress_data[task_id] = {
        'current': 0,
        'total': 0,
        'status': 'Starting...',
        'products': [],
        'ai_success': 0,
        'ai_failed': 0,
        'ai_retry': [],
        'lock': threading.Lock(),
        'ai_lock': threading.Lock()
    }
    
    threading.Thread(target=background_task, args=(po_file_content, po_file.filename, task_id), daemon=True).start()
    
    return jsonify({'task_id': task_id}), 202

@app.route('/progress/<task_id>')
def progress(task_id):
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
        
    with progress_data[task_id]['lock']:
        data_copy = {k: v for k, v in progress_data[task_id].items() if k not in ['lock', 'ai_lock', 'ai_retry']}
        data_copy['ai_pending'] = len(progress_data[task_id]['ai_retry'])
    
    return jsonify(data_copy)

@app.route('/download-all/<task_id>', methods=['GET'])
def download_all_files(task_id):
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
        
    try:
        memory_zip = BytesIO()
        
        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            with progress_data[task_id]['lock']:
                products = progress_data[task_id]['products'].copy()
            
            for product in products:
                if product['filename'] in product_data_store:
                    zf.writestr(product['filename'], product_data_store[product['filename']])
        
        memory_zip.seek(0)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        zip_filename = f"shipping_files_{timestamp}.zip"
        
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    
    except Exception as e:
        logger.error(f"Error creating zip file: {str(e)}")
        return jsonify({'error': 'Failed to create zip file'}), 500
    
@app.route('/download_main_products_csv')
def download_main_products_csv():
    global latest_products, from_info
    try:
        if not latest_products:
            return jsonify({'error': 'No product data available. Please upload and process a PO file first.'}), 404

        headers = [
            'FromName', 'FromCompany', 'FromStreet', 'FromStreet2', 'FromCity', 'FromState',
            'FromZip', 'FromPhone', 'ToName', 'ToCompany', 'ToStreet', 'ToStreet2',
            'ToCity', 'ToState', 'ToZip', 'ToPhone', 'Weight', 'Length', 'Width',
            'Height', 'Description', 'order num', 'Reference2', 'Signature'
        ]

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        
        for product in latest_products:
            row = [
                from_info['FromName'], from_info['FromCompany'], from_info['FromStreet'],
                from_info['FromStreet2'], from_info['FromCity'], from_info['FromState'],
                from_info['FromZip'], from_info['FromPhone'],
                product['to_info']['ToName'],
                product['to_info']['ToCompany'],
                product['to_info']['ToStreet'],
                product['to_info']['ToStreet2'],
                product['to_info']['ToCity'],
                product['to_info']['ToState'],
                product['to_info']['ToZip'],
                product['to_info']['ToPhone'],
                product['weight'],
                product['length'],
                product['width'],
                product['height'],
                product['short_description'],
                product['order_number'],
                product['po_number'],
                ''
            ]
            writer.writerow(row)

        output.seek(0)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return send_file(
            BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'all_products_{timestamp}.csv'
        )
    except Exception as e:
        logger.error(f"Error creating main products CSV: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_unique_products_zip')
def download_unique_products_zip():
    global latest_products, from_info
    try:
        if not latest_products:
            return jsonify({'error': 'No product data available. Please upload and process a PO file first.'}), 404

        from collections import defaultdict
        product_groups = defaultdict(list)
        for product in latest_products:
            desc = product.get('description') or product.get('product') or 'Unknown Product'
            product_groups[desc].append(product)

        headers = [
            'FromName', 'FromCompany', 'FromStreet', 'FromStreet2', 'FromCity', 'FromState',
            'FromZip', 'FromPhone', 'ToName', 'ToCompany', 'ToStreet', 'ToStreet2',
            'ToCity', 'ToState', 'ToZip', 'ToPhone', 'Weight', 'Length', 'Width',
            'Height', 'Description', 'order num', 'Reference2', 'Signature'
        ]

        memory_zip = BytesIO()
        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for desc, group in product_groups.items():
                safe_desc = re.sub(r'[^\w\s-]', '', str(desc))[:50].strip().replace(' ', '_')
                filename = f"{safe_desc or 'Unknown_Product'}.csv"
                csv_content = generate_group_csv(group, headers)
                zf.writestr(filename, csv_content)

        memory_zip.seek(0)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'unique_products_{timestamp}.zip'
        )
    except Exception as e:
        logger.error(f"Error creating ZIP for unique products: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    if filename in product_data_store:
        return send_file(
            BytesIO(product_data_store[filename]),
            as_attachment=True,
            mimetype='text/csv',
            download_name=filename
        )
    return jsonify({'error': 'File not found'}), 404

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        global product_data_store
        count = len(product_data_store)
        product_data_store = {}
        return jsonify({'success': True, 'count': count})
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)