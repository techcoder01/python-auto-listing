from flask import Flask, request, jsonify, send_file, render_template, Response
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
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

uploaded_cloudinary_ids = set()

def upload_file_to_cloudinary(file_storage):
    result = cloudinary.uploader.upload(
        file_storage,
        resource_type="raw"
    )
    uploaded_cloudinary_ids.add(result['public_id'])
    return result['secure_url'], result['public_id']

def download_file_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PRODUCTS_FOLDER'] = 'products'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# Enable CORS
from flask_cors import CORS
CORS(app)

# Configure Gemini API
API_KEY = 'AIzaSyDhzCeMj7YBXh7suNKdOzyGUtnc1KWM3fc'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')
global from_info  # <-- Declare here because you assign to from_info


# Add these with your other global variables
stop_execution_flag = False
stop_execution_lock = threading.Lock()
active_threads = {}  # Track active processing threads
threads_lock = threading.Lock()  # Lock for thread dictionary

@app.route('/stop_execution', methods=['POST'])
def stop_execution():
    """Endpoint to stop all Gemini API processing"""
    global stop_execution_flag
    
    with stop_execution_lock:
        stop_execution_flag = True
        
    # Cancel any ongoing tasks
    with threads_lock:
        for task_id, thread_info in list(active_threads.items()):
            if thread_info['thread'].is_alive():
                # Mark the task as cancelled in progress data
                with progress_data[task_id]['lock']:
                    progress_data[task_id]['error'] = 'Processing stopped by user'
                    progress_data[task_id]['status'] = 'Cancelled'
                
                # Send stop event
                with sse_manager.lock:
                    sse_manager.send_event(task_id, 'stop', {'message': 'Execution stopped by user'})
    
    return jsonify({'success': True, 'message': 'Stopping all execution'})

# Global from_info with default empty values
from_info = {
    'FromName': '',
    'FromCompany': '',
    'FromStreet': '',
    'FromStreet2': '',
    'FromCity': '',
    'FromState': '',
    'FromZip': '',
    'FromPhone': ''
}
def extract_from_info(filepath):
    """Extract from information from the provided file."""
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Get the first row of the file to extract "from" info
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
            product_data['short_description'],  # <-- Use short description here!
            product_data['order_number'],
            product_data['po_number'],
            ''
        ]
        writer.writerow(row)
    return output.getvalue()


@app.route('/download_unique_products_zip')
def download_unique_products_zip():
    global latest_products, from_info  # <-- Add here for clarity
    try:
        global latest_products
        if not latest_products:
            return jsonify({'error': 'No product data available. Please upload and process a PO file first.'}), 404

        # Group products by their original description
        from collections import defaultdict
        product_groups = defaultdict(list)
        for product in latest_products:
            # Find the original description for grouping
            desc = product.get('description') or product.get('product') or 'Unknown Product'
            product_groups[desc].append(product)

        # Use your required headers
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
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name='unique_products.zip'
        )
    except Exception as e:
        logger.error(f"Error creating ZIP for unique products: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


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
            
            self.last_request_time = time.time()
            self.request_count += 1
            return current_time

gemini_limiter = StrictRateLimiter(rate_per_minute=60)

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

def shorten_description(description, task_id, is_retry=False):
    global stop_execution_flag
    
    # Check if we should stop - quick check first without lock
    if stop_execution_flag:
        with stop_execution_lock:
            if stop_execution_flag:
                raise Exception("Processing stopped by user")
    
    if len(description) <= 30:
        return description
        
    try:
        if not is_retry:
            # Check stop flag again before rate limiting
            if stop_execution_flag:
                with stop_execution_lock:
                    if stop_execution_flag:
                        raise Exception("Processing stopped by user")
            
            gemini_limiter.wait()
        
        # Final check before making API call
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
        # Only log if this wasn't a stop request
        with stop_execution_lock:
            if not stop_execution_flag:
                logger.error(f"Error shortening description: {str(e)}")
        return None

@app.route('/reset_execution_flag', methods=['POST'])
def reset_execution_flag():
    """Endpoint to reset the stop flag (allows new processes to run)"""
    global stop_execution_flag
    
    with stop_execution_lock:
        stop_execution_flag = False
        
    # Clean up any completed threads
    with threads_lock:
        for task_id in list(active_threads.keys()):
            if not active_threads[task_id]['thread'].is_alive():
                del active_threads[task_id]
        
    return jsonify({'success': True, 'message': 'Execution flag reset'})

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
        
        # Use the shorter filename function
        filename = generate_shorter_filename(product_data)
        filepath = os.path.join(app.config['PRODUCTS_FOLDER'], filename)
        
        with open(filepath, 'w', newline='') as f:
            f.write(csv_content)
        
        return {
            'product': product_data['short_description'],
            'filename': filename,
            'download_url': f'/download/{filename}',
            'to_info': product_data['to_info']
        }, None
    
    except Exception as e:
        logger.error(f"Error processing product: {str(e)}")
        return None, product_data
    
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
    
    return output.getvalue()

# Update the process_po_file function to store original data
def process_po_file(filepath, task_id):
    try:
        if filepath.endswith('.csv'):
            chunksize = 1000
            df = pd.concat([chunk for chunk in pd.read_csv(filepath, chunksize=chunksize)])
        else:
            df = pd.read_excel(filepath)
        
        # Store the original data for consolidation later
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
        
        # Store the original PO data for consolidation
        with progress_data[task_id]['lock']:
            progress_data[task_id]['po_data'] = original_data

            # In your process_po_file function, after storing in progress_data:

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



def background_task(po_url, from_url, task_id):
    global stop_execution_flag
    
    # Register this thread
    with threads_lock:
        active_threads[task_id] = {
            'thread': threading.current_thread(),
            'start_time': time.time()
        }
    
    try:
        po_file_obj = download_file_from_url(po_url)
        from_file_obj = download_file_from_url(from_url)
        # Now use po_file_obj and from_file_obj with pandas
        if po_url.lower().endswith('.csv'):
            po_df = pd.read_csv(po_file_obj)
        else:
            po_df = pd.read_excel(po_file_obj)
        # Check if we should stop before starting
        with stop_execution_lock:
            if stop_execution_flag:
                with progress_data[task_id]['lock']:
                    progress_data[task_id]['error'] = 'Processing stopped before starting'
                    progress_data[task_id]['status'] = 'Cancelled'
                send_progress_update(task_id)
                return
            
        results = process_po_file(filepath, task_id)
        if isinstance(results, dict) and 'error' in results:
            with progress_data[task_id]['lock']:
                progress_data[task_id]['error'] = results['error']
            send_progress_update(task_id)
            return
        
        with progress_data[task_id]['lock']:
            progress_data[task_id]['total'] = len(results)
            progress_data[task_id]['ai_retry'] = []
        send_progress_update(task_id)
        
        # Process with careful rate limiting
        max_workers = 4  # Conservative number to avoid rate limits
        processed_count = 0
        retry_items = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # First pass - try all items
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
            
            # Retry failed items with more conservative approach
            max_retries = 5
            retry_count = 0
            
            while retry_items and retry_count < max_retries:
                retry_count += 1
                logger.info(f"Retry attempt {retry_count} with {len(retry_items)} items")
                
                # Process retries one at a time to strictly respect rate limits
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
                    
                    # Small delay between retries
                    time.sleep(0.5)
                
                retry_items = new_retries
                
                if not retry_items:
                    break
                
                # If we still have retries, wait before next attempt
                if retry_items and retry_count < max_retries:
                    wait_time = min(10, 2 ** retry_count)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before next retry")
                    time.sleep(wait_time)
            
            # Final fallback for any remaining items
            if retry_items:
                logger.warning(f"Falling back to simple shortening for {len(retry_items)} items")
                
                for retry_data in retry_items:
                    try:
                        short_desc = simple_shorten(retry_data['description'])
                        retry_data['short_description'] = f"{short_desc} (Qty: {retry_data['qty']})"
                        csv_content = generate_product_csv(retry_data)
                        
                        clean_desc = ''.join(c if c.isalnum() else '_' for c in retry_data['short_description'])[:40]
                        filename = f"product_{retry_data['po_number']}_{retry_data['order_number']}_{clean_desc}_{uuid.uuid4().hex[:8]}.csv"
                        filepath = os.path.join(app.config['PRODUCTS_FOLDER'], filename)
                        
                        with open(filepath, 'w', newline='') as f:
                            f.write(csv_content)
                        
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
        
        # Only mark complete if all items processed
        with progress_data[task_id]['lock']:
            if processed_count == progress_data[task_id]['total']:
                progress_data[task_id]['status'] = 'Completed'
                progress_data[task_id]['success'] = True
            else:
                progress_data[task_id]['status'] = 'Failed - Not all items processed'
                progress_data[task_id]['error'] = 'Could not process all items'
        
        send_progress_update(task_id)
        global latest_products  # <-- Declare here before assigning
        latest_products = results  # or however you set it
        
    except Exception as e:
        # Check if this was a stop request
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
        # Clean up thread tracking
        with threads_lock:
            if task_id in active_threads:
                del active_threads[task_id]
        
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

@app.route('/clean_cloudinary', methods=['POST'])
def clean_cloudinary():
    deleted = []
    errors = []
    for public_id in list(uploaded_cloudinary_ids):
        try:
            cloudinary.uploader.destroy(public_id, resource_type="raw")
            deleted.append(public_id)
            uploaded_cloudinary_ids.remove(public_id)
        except Exception as e:
            errors.append({'public_id': public_id, 'error': str(e)})
    return jsonify({'deleted': deleted, 'errors': errors})


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

def generate_shorter_filename(product_data):
    """Generate a shorter, more concise filename for the CSV file."""
    # Extract PO number and order number (use last 5 digits if they're long)
    po_num = product_data['po_number']
    order_num = product_data['order_number']
    
    # Get a clean, short version of the product description
    if 'short_description' in product_data:
        desc = product_data['short_description']
        # Remove (Qty: X) and (See Details) from description for filename
        desc = re.sub(r'\s*\(Qty:.*?\)', '', desc)
        desc = re.sub(r'\s*\(See Details\)', '', desc)
    else:
        desc = product_data.get('description', '')
    
    # Limit description to 3-4 words max for shorter filename
    words = re.findall(r'\w+', desc)
    short_desc = ' '.join(words[:3])
    
    # Get customer name (just first part)
    customer_name = product_data['to_info']['ToName'].split()[0] if product_data['to_info']['ToName'] else 'unknown'
    
    # Clean up special characters
    clean_text = re.sub(r'[^\w\s]', '', f"{short_desc}_{customer_name}")
    clean_text = re.sub(r'\s+', '_', clean_text)
    
    # Create unique but short identifier
    short_uuid = uuid.uuid4().hex[:6]
    
    # Build the filename (limited to reasonable length)
    filename = f"prod_{clean_text}_{short_uuid}.csv"
    
    return filename

def cleanup_existing_files():
    """Clean up all existing files in the products folder."""
    try:
        folder = app.config['PRODUCTS_FOLDER']
        count = 0
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    count += 1
        logger.info(f"Cleaned up {count} existing files before starting new process")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up existing files: {str(e)}")
        return 0

# Add this new route to your Flask app
@app.route('/update_api_key', methods=['POST'])
def update_api_key():
    try:
        data = request.get_json()
        new_key = data.get('api_key', '').strip()
        
        if not new_key:
            return jsonify({'error': 'API key cannot be empty'}), 400
            
        # Update the global API key configuration
        genai.configure(api_key=new_key)
        
        return jsonify({'success': True, 'message': 'API key updated successfully'})
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    # Clean up existing files first
    cleanup_existing_files()

    # Get API key from form if provided
    user_api_key = request.form.get('gemini_api_key', '').strip()
    if user_api_key:
        try:
            genai.configure(api_key=user_api_key)
            logger.info("Using user-provided API key")
        except Exception as e:
            logger.error(f"Error configuring with user API key: {str(e)}")
            return jsonify({'error': 'Invalid API key provided'}), 400
    
    # Check for PO file
    if 'po_file' not in request.files:
        return jsonify({'error': 'No PO file provided'}), 400
    
    po_file = request.files['po_file']
    
    if po_file.filename == '':
        return jsonify({'error': 'No PO file selected'}), 400
    
    if not allowed_file(po_file.filename):
        return jsonify({'error': 'PO file type not allowed'}), 400
    
    # Check for From file
    if 'from_file' not in request.files:
        return jsonify({'error': 'No From information file provided'}), 400
    
    from_file = request.files['from_file']
    
    if from_file.filename == '':
        return jsonify({'error': 'No From information file selected'}), 400
    
    if not allowed_file(from_file.filename):
        return jsonify({'error': 'From file type not allowed'}), 400
    
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRODUCTS_FOLDER'], exist_ok=True)
    
    # Save From file and extract information
    from_filename = secure_filename(from_file.filename)
    from_filepath = os.path.join(app.config['UPLOAD_FOLDER'], from_filename)
    from_file.save(from_filepath)
    
    # Extract from information
    if not extract_from_info(from_filepath):
        try:
            os.remove(from_filepath)
        except Exception as e:
            logger.error(f"Error removing from file: {str(e)}")
        return jsonify({'error': 'Failed to extract from information'}), 400
    
    # Clean up from file
    try:
        os.remove(from_filepath)
    except Exception as e:
        logger.error(f"Error removing from file: {str(e)}")
    
    # Save PO file
    po_url, po_id = upload_file_to_cloudinary(po_file)
    from_url, from_id = upload_file_to_cloudinary(from_file)

    # Store Cloudinary URLs in progress_data or session as needed
    task_id = str(uuid.uuid4())
    progress_data[task_id] = {
        'po_url': po_url,
        'from_url': from_url,
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
    
    threading.Thread(target=background_task, args=(po_url, from_url, task_id), daemon=True).start()

    return jsonify({'task_id': task_id}), 202    

@app.route('/progress/<task_id>')
def progress(task_id):
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
        
    with progress_data[task_id]['lock']:
        data_copy = {k: v for k, v in progress_data[task_id].items() if k not in ['lock', 'ai_lock', 'ai_retry']}
        data_copy['ai_pending'] = len(progress_data[task_id]['ai_retry'])
    
    return jsonify(data_copy)

import zipfile
import io

@app.route('/download-all/<task_id>', methods=['GET'])
def download_all_files(task_id):
    """Create a zip file with all generated files for a task and send it."""
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
        
    try:
        # Create an in-memory zip file
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Get all product files for this task
            with progress_data[task_id]['lock']:
                products = progress_data[task_id]['products'].copy()
            
            # Add each file to the zip
            for product in products:
                filepath = os.path.join(app.config['PRODUCTS_FOLDER'], product['filename'])
                if os.path.exists(filepath):
                    # Read the file content
                    with open(filepath, 'r') as f:
                        file_content = f.read()
                    
                    # Add to zip with the same filename
                    zf.writestr(product['filename'], file_content)
        
        # Prepare response
        memory_file.seek(0)
        
        # Generate a filename for the zip based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        zip_filename = f"shipping_files_{timestamp}.zip"
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    
    except Exception as e:
        logger.error(f"Error creating zip file: {str(e)}")
        return jsonify({'error': 'Failed to create zip file'}), 500
    
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(app.config['PRODUCTS_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(
            filepath,
            as_attachment=True,
            mimetype='text/csv',
            download_name=filename
        )
    return jsonify({'error': 'File not found'}), 404

@app.route('/download_main_products_csv')
def download_main_products_csv():
    try:
        global latest_products, from_info
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

        for product_data in latest_products:
            row = [
                from_info.get('FromName', ''),
                from_info.get('FromCompany', ''),
                from_info.get('FromStreet', ''),
                from_info.get('FromStreet2', ''),
                from_info.get('FromCity', ''),
                from_info.get('FromState', ''),
                from_info.get('FromZip', ''),
                from_info.get('FromPhone', ''),
                product_data['to_info'].get('ToName', ''),
                product_data['to_info'].get('ToCompany', ''),
                product_data['to_info'].get('ToStreet', ''),
                product_data['to_info'].get('ToStreet2', ''),
                product_data['to_info'].get('ToCity', ''),
                product_data['to_info'].get('ToState', ''),
                product_data['to_info'].get('ToZip', ''),
                product_data['to_info'].get('ToPhone', ''),
                product_data.get('weight', ''),
                product_data.get('length', ''),
                product_data.get('width', ''),
                product_data.get('height', ''),
                product_data.get('short_description', ''),  # Use shortened description
                product_data.get('order_number', ''),
                product_data.get('po_number', ''),
                ''  # Signature empty
            ]
            writer.writerow(row)

        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='all_products.csv'
        )
    except Exception as e:
        logger.error(f"Error creating main products CSV: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        count = 0
        for filename in os.listdir(app.config['PRODUCTS_FOLDER']):
            file_path = os.path.join(app.config['PRODUCTS_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                count += 1
        return jsonify({'success': True, 'count': count})
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PRODUCTS_FOLDER'], exist_ok=True)
    app.run(debug=True, threaded=True)