from flask import Flask, request, jsonify, send_file, render_template, Response
from werkzeug.utils import secure_filename
import os
import pandas as pd
import uuid
import csv
from io import StringIO
import numpy as np
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import json
import logging
import queue
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PRODUCTS_FOLDER'] = 'products'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}

# Enable CORS
from flask_cors import CORS
CORS(app)

# Configure Gemini API
API_KEY = 'AIzaSyCrpHo3kccVcJr6cIrpuQH7pLE0ExQwV54'
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

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


def batch_generate_short_descriptions(products):
    """Generates individual short descriptions for all products in a single API call."""
    if not products:
        return {}  # Return empty dict if no products provided
    
    # Create a combined prompt for all products
    product_prompts = []
    for i, product in enumerate(products):
        description = product.get('description', '')
        if description:
            product_prompts.append(f"Product {i+1}: {description}")
    
    if not product_prompts:
        # Default description if no descriptions available
        return {i: "Assorted Item" for i in range(len(products))}
    
    combined_prompt = """Shorten the following product descriptions to a maximum of 8-15 words each, keeping the essential details like:
    - Product name
    - Form (e.g., gummies)
    - Key ingredient (e.g., creatine monohydrate) and its dosage
    - Primary benefits
    - Ensure the exact flavor (especially Blue Raspberry) is mentioned when available.
    - Remove gender-specific terms like "Man", "Women", "Male", or "Female" completely.

    The product descriptions are:
    """
    combined_prompt += "\n\n".join(product_prompts)
    combined_prompt += "\n\nOutput only a JSON array of the shortened descriptions."

    try:
        # Single API call for all descriptions
        gemini_limiter.wait()
        response = model.generate_content(combined_prompt, temperature=0.9)  # Adjusted temperature here
        
        # Parse the JSON response
        import json
        try:
            descriptions = json.loads(response.text)
            # Ensure we have a description for each product
            if len(descriptions) != len(products):
                logger.warning(f"Mismatch in description count: got {len(descriptions)}, expected {len(products)}")
                # Pad or truncate as needed
                if len(descriptions) < len(products):
                    descriptions.extend(["Product Item"] * (len(products) - len(descriptions)))
                else:
                    descriptions = descriptions[:len(products)]
            
            # Create a dictionary mapping product index to description
            return {i: desc.strip('"\'') + " (See Details)" for i, desc in enumerate(descriptions)}
        
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from AI")
            # Fallback to basic descriptions
            return {i: "Product Item (See Details)" for i in range(len(products))}
    
    except Exception as e:
        logger.error(f"Error in batch generating descriptions: {str(e)}")
        return {i: "General Product (See Details)" for i in range(len(products))}



def process_product(product_data, task_id, is_retry=False):
    try:
        if not is_retry:
            with progress_data[task_id]['lock']:
                progress_data[task_id]['current'] += 1
                current = progress_data[task_id]['current']
                total = progress_data[task_id]['total']
                progress_data[task_id]['status'] = f"Processing {current}/{total}"
        
        if len(product_data.get('description', '')) > 0:
            # Use simple_shorten instead, which already exists in your code
            short_desc = simple_shorten(product_data['description'])
            
            if short_desc is None:
                return None, product_data
        else:
            short_desc = "Unknown Product"
            
        product_data['short_description'] = f"{short_desc} (Qty: {product_data['qty']})"
        csv_content = generate_product_csv(product_data)
        
        clean_desc = ''.join(c if c.isalnum() else '_' for c in product_data['short_description'])[:40]
        filename = f"product_{product_data['po_number']}_{product_data['order_number']}_{clean_desc}_{uuid.uuid4().hex[:8]}.csv"
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
        '', '', '', '', '', '', '', '',
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

def process_po_file(filepath, task_id):
    try:
        if filepath.endswith('.csv'):
            chunksize = 1000
            df = pd.concat([chunk for chunk in pd.read_csv(filepath, chunksize=chunksize)])
        else:
            df = pd.read_excel(filepath)
        
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

def background_task(filepath, task_id):
    try:
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
        
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")
        with progress_data[task_id]['lock']:
            progress_data[task_id]['error'] = str(e)
        send_progress_update(task_id)
    finally:
        try:
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['PRODUCTS_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
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
        
        threading.Thread(target=background_task, args=(filepath, task_id), daemon=True).start()
        
        return jsonify({'task_id': task_id}), 202
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/progress/<task_id>')
def progress(task_id):
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
        
    with progress_data[task_id]['lock']:
        data_copy = {k: v for k, v in progress_data[task_id].items() if k not in ['lock', 'ai_lock', 'ai_retry']}
        data_copy['ai_pending'] = len(progress_data[task_id]['ai_retry'])
    
    return jsonify(data_copy)

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