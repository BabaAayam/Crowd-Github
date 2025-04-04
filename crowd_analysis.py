import time
import cv2
import numpy as np
import tensorflow as tf
import requests
import json
import zlib
import os
import csv
import base64
from threading import Thread
from datetime import datetime

class CrowdAnalyzer:
    def __init__(self, model_path="models/ssd_mobilenet.tflite", 
                 server_url="http://localhost:5000/receive_data", 
                 edge_mode=False,
                 advanced_analytics_interval=3):
        # Model initialization
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=4  # Use all CPU cores on Pi
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        
        # Server configuration
        self.server_url = server_url
        self.edge_mode = edge_mode
        self.frame_counter = 0
        self.last_send_time = time.time()
        self.last_send_status = False
        self.advanced_analytics_interval = advanced_analytics_interval
        
        # Data tracking and CSV offline cache
        self.people_counts = []
        self.anomalies_log = []
        self.processing_times = []
        self.cache_file = "crowd_cache.csv"
        
        # Initialize CSV file with headers if not exists
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "count", "detections", "frame_data"])

    def process_frame(self, frame):
        try:
            # Frame skipping for Pi optimization
            self.frame_counter += 1
            if self.frame_counter % 3 != 0:  # Process every 3rd frame
                return frame, 0, [], []

            start_time = time.time()

            # Preprocessing
            resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

            # Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()

            # Process results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

            # Detection processing
            count = 0
            detections = []
            anomalies = []
            h, w = frame.shape[:2]

            for i in range(num_detections):
                if scores[i] > 0.5 and classes[i] == 0:  # Class 0 = person
                    ymin, xmin, ymax, xmax = boxes[i]
                    x1, y1 = int(xmin * w), int(ymin * h)
                    x2, y2 = int(xmax * w), int(ymax * h)
                    detections.append((x1, y1, x2, y2))
                    count += 1
                    if count > 10:  # Anomaly threshold
                        anomalies.append((x1, y1, x2, y2))

            # Metrics logging
            processing_time = round((time.time() - start_time) * 1000, 2)
            self.people_counts.append(count)
            self.anomalies_log.append(len(anomalies))
            self.processing_times.append(processing_time)

            # Prepare payload for server
            payload = {
                "count": count,
                "detections": detections,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time
            }

            # Add full frame for advanced analytics at specified interval
            if self.frame_counter % self.advanced_analytics_interval == 0:
                # Compress frame to JPEG (quality 70 to reduce size)
                _, jpeg_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                payload["full_frame"] = base64.b64encode(jpeg_frame).decode('ascii')
                payload["frame_id"] = self.frame_counter

            # Conditional HTTP send
            current_time = time.time()
            if count > 10 or (current_time - self.last_send_time) > 30:
                Thread(target=self.send_data_to_server, args=(payload,)).start()
                self.last_send_time = current_time

            return frame, count, detections, anomalies

        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
            return frame, 0, [], []

    def send_data_to_server(self, payload):
        """Send data with CSV fallback (modified for advanced analytics)"""
        try:
            # Compress payload
            compressed = zlib.compress(json.dumps(payload).encode())
            
            # Determine content type
            headers = {
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "X-Data-Type": "advanced" if "full_frame" in payload else "basic"
            }
            
            # HTTP POST
            response = requests.post(
                self.server_url,
                data=compressed,
                headers=headers,
                timeout=2
            )
            self.last_send_status = response.status_code == 200
            
            if not self.last_send_status:
                self.cache_to_csv(payload)  # Fallback to CSV
                
        except Exception as e:
            print(f"[ERROR] HTTP send failed: {e}")
            self.cache_to_csv(payload)

    def cache_to_csv(self, payload):
        """Append data to CSV file (modified for frame data)"""
        try:
            with open(self.cache_file, 'a', newline='') as f:
                writer = csv.writer(f)
                frame_data = payload.get("full_frame", "null")
                writer.writerow([
                    payload["timestamp"],
                    payload["count"],
                    json.dumps(payload["detections"]),
                    frame_data[:30] + "..." if isinstance(frame_data, str) else "null"
                ])
        except Exception as e:
            print(f"[WARNING] CSV write failed: {e}")

    def __del__(self):
        """Ensure all data is flushed to CSV on exit"""
        pass  # CSV writes are immediate