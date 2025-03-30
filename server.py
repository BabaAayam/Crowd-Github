import os
import zlib
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS if you have a separate frontend

# Database setup
DB_NAME = 'crowd_data.db'
CSV_SYNC_INTERVAL = 300  # 5 minutes for CSV sync

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Main crowd data table
    c.execute('''CREATE TABLE IF NOT EXISTS crowd_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  device_id TEXT,
                  count INTEGER,
                  detections TEXT,  # JSON string of detection coordinates
                  timestamp DATETIME,
                  processing_time REAL)''')
    
    # Device metadata table
    c.execute('''CREATE TABLE IF NOT EXISTS devices
                 (device_id TEXT PRIMARY KEY,
                  last_seen DATETIME,
                  ip_address TEXT,
                  location TEXT)''')
    
    # Anomaly logs table
    c.execute('''CREATE TABLE IF NOT EXISTS anomalies
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  device_id TEXT,
                  count INTEGER,
                  timestamp DATETIME,
                  FOREIGN KEY(device_id) REFERENCES devices(device_id))''')
    
    conn.commit()
    conn.close()

init_db()

@app.route('/receive_data', methods=['POST'])
def receive_data():
    """Endpoint for edge devices to send crowd data"""
    try:
        # Decompress the data
        compressed_data = request.data
        json_data = zlib.decompress(compressed_data).decode('utf-8')
        data = json.loads(json_data)
        
        # Validate required fields
        if 'count' not in data or 'timestamp' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Extract device ID from headers or data
        device_id = request.headers.get('X-Device-ID', 'default_device')
        ip_address = request.remote_addr
        
        # Store in database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Insert crowd data
        c.execute('''INSERT INTO crowd_data 
                     (device_id, count, detections, timestamp)
                     VALUES (?, ?, ?, ?)''',
                  (device_id, 
                   data['count'],
                   json.dumps(data.get('detections', [])),
                   data['timestamp']))
        
        # Update device info
        c.execute('''INSERT OR REPLACE INTO devices 
                     (device_id, last_seen, ip_address)
                     VALUES (?, ?, ?)''',
                  (device_id, datetime.now().isoformat(), ip_address))
        
        # Log anomalies if count exceeds threshold
        if data['count'] > 10:  # Same threshold as edge device
            c.execute('''INSERT INTO anomalies 
                         (device_id, count, timestamp)
                         VALUES (?, ?, ?)''',
                      (device_id, data['count'], data['timestamp']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_crowd_data():
    """Retrieve crowd data for visualization"""
    try:
        device_id = request.args.get('device_id', 'default_device')
        hours = int(request.args.get('hours', 24))
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Get recent crowd data
        c.execute('''SELECT timestamp, count, detections 
                     FROM crowd_data 
                     WHERE device_id = ? 
                     AND timestamp >= datetime('now', ?)
                     ORDER BY timestamp DESC''',
                  (device_id, f'-{hours} hours'))
        rows = c.fetchall()
        
        # Format data for response
        data = [{
            'timestamp': row[0],
            'count': row[1],
            'detections': json.loads(row[2]) if row[2] else []
        } for row in rows]
        
        # Get summary statistics
        c.execute('''SELECT 
                     AVG(count), MAX(count), MIN(count), COUNT(*) 
                     FROM crowd_data 
                     WHERE device_id = ? 
                     AND timestamp >= datetime('now', ?)''',
                  (device_id, f'-{hours} hours'))
        stats = c.fetchone()
        
        conn.close()
        
        return jsonify({
            'data': data,
            'stats': {
                'average': stats[0],
                'max': stats[1],
                'min': stats[2],
                'samples': stats[3]
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plot', methods=['GET'])
def get_plot():
    """Generate and return a crowd count plot"""
    try:
        device_id = request.args.get('device_id', 'default_device')
        hours = int(request.args.get('hours', 24))
        
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        c.execute('''SELECT timestamp, count
                     FROM crowd_data
                     WHERE device_id = ?
                     AND timestamp >= datetime('now', ?)
                     ORDER BY timestamp''',
                  (device_id, f'-{hours} hours'))
        data = c.fetchall()
        conn.close()
        
        if not data:
            return jsonify({'error': 'No data available'}), 404
        
        timestamps = [row[0] for row in data]
        counts = [row[1] for row in data]
        
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, counts, 'b-')
        plt.title(f'Crowd Count - Last {hours} Hours')
        plt.xlabel('Time')
        plt.ylabel('People Count')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return as base64 encoded image
        return jsonify({
            'image': base64.b64encode(buf.read()).decode('utf-8')
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sync_csv', methods=['POST'])
def sync_csv():
    """Endpoint for edge devices to sync their CSV backup data"""
    try:
        device_id = request.headers.get('X-Device-ID', 'default_device')
        csv_file = request.files['csv_file']
        
        # Process CSV and insert into DB
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        for line in csv_file:
            timestamp, count, detections = line.decode('utf-8').strip().split(',')
            c.execute('''INSERT OR IGNORE INTO crowd_data
                         (device_id, count, detections, timestamp)
                         VALUES (?, ?, ?, ?)''',
                      (device_id, int(count), detections, timestamp))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'rows_synced': 'variable'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    """Simple health check endpoint"""
    return jsonify({'status': 'alive'}), 200

def csv_sync_worker():
    """Background worker to periodically sync CSV data from edge devices"""
    while True:
        time.sleep(CSV_SYNC_INTERVAL)
        # Implementation would check for CSV files and sync them
        print("CSV sync check...")

# Start CSV sync thread
threading.Thread(target=csv_sync_worker, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)