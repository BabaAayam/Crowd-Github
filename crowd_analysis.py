import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import csv
import json
import zlib
import base64
import requests
from threading import Thread, Lock
from datetime import datetime
from collections import deque
import logging
import platform
import traceback

class CrowdAnalyzer:
    def __init__(self, model_path="models/ssd_mobilenet.tflite", 
                 server_url="http://192.168.191.1:5000/receive_data",  # Update with your Windows server IP
                 edge_mode=False,
                 advanced_analytics_interval=3):
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Check if running on Raspberry Pi
        self.is_pi = False
        try:
            if platform.system() == 'Linux':
                with open('/proc/device-tree/model', 'r') as f:
                    self.is_pi = 'raspberry pi' in f.read().lower()
        except:
            self.is_pi = platform.machine() in ('armv7l', 'aarch64')
        
        # Model initialization
        try:
            # Try to use Coral TPU if available
            try:
                if self.is_pi:
                    self.interpreter = tflite.Interpreter(
                        model_path=model_path,
                        num_threads=4,  # Use all CPU cores
                        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]  # Use Coral TPU if available
                    )
                else:
                    self.interpreter = tflite.Interpreter(
                        model_path=model_path,
                        num_threads=4
                    )
            except Exception as e:
                self.logger.warning(f"Coral TPU not available, using CPU: {str(e)}")
                self.interpreter = tflite.Interpreter(
                    model_path=model_path,
                    num_threads=4
                )
            
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get model's expected input dimensions
            _, self.model_height, self.model_width, _ = self.input_details[0]['shape']
            
            self.logger.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Performance optimizations
        self.process_every = 4 if self.is_pi else 2  # Process more frames for normal speed
        self.detection_thresh = 0.4  # Keep threshold low for smooth detection
        self.frame_buffer = None
        self.frame_lock = Lock()
        self.last_detections = []
        self.last_count = 0
        self.last_frame = None
        self.last_processed_frame = None
        self.frame_history = deque(maxlen=2)  # Reduced history for normal speed
        
        # Server configuration
        self.server_url = server_url
        self.edge_mode = edge_mode
        self.frame_counter = 0
        self.last_send_time = time.time()
        self.advanced_analytics_interval = advanced_analytics_interval
        self.send_thread = None
        self.send_lock = Lock()
        
        # Data tracking
        self.people_counts = deque(maxlen=15)
        self.anomalies_log = deque(maxlen=15)
        self.processing_times = deque(maxlen=15)
        self.cache_file = "crowd_cache.csv"
        
        # Initialize cache file
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "count", "detections", "frame_data", "processing_time_ms"])

        # Pre-allocate numpy arrays for faster processing
        self.input_shape = (1, self.model_height, self.model_width, 3)
        self.input_tensor = np.zeros(self.input_shape, dtype=np.uint8)
        
        # Pre-allocate frame processing arrays
        self.frame_shape = (480, 640, 3)  # Standard frame size
        self.processed_frame = np.zeros(self.frame_shape, dtype=np.uint8)

    def process_frame(self, frame):
        """Process a single frame with improved performance and interpolation"""
        try:
            self.frame_counter += 1
            
            # Add frame to history
            self.frame_history.append(frame.copy())
            
            # Skip frames for performance
            if self.frame_counter % self.process_every != 0:
                # Use interpolation for skipped frames
                if len(self.frame_history) >= 2 and self.last_processed_frame is not None:
                    # Simple linear interpolation between last processed frame and current frame
                    alpha = (self.frame_counter % self.process_every) / self.process_every
                    interpolated_frame = cv2.addWeighted(
                        self.last_processed_frame, 1 - alpha,
                        frame, alpha,
                        0
                    )
                    return self._draw_cached_results(interpolated_frame)
                return self._draw_cached_results(frame)

            start_time = time.time()
            
            # Cache current frame
            self.last_frame = frame.copy()

            # Resize frame (more efficient)
            frame_resized = cv2.resize(frame, (self.model_width, self.model_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # Update pre-allocated input tensor
            self.input_tensor[0] = frame_resized
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], self.input_tensor)
            self.interpreter.invoke()
            
            # Get results
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Process detections (optimized)
            detections = []
            count = 0
            anomalies = []
            
            # Get frame dimensions once
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / self.model_width
            scale_y = frame_h / self.model_height
            
            # Process only high confidence detections
            high_conf_idx = scores > self.detection_thresh
            boxes = boxes[high_conf_idx]
            classes = classes[high_conf_idx]
            
            # Vectorized box processing
            person_boxes = boxes[classes == 0]
            if len(person_boxes) > 0:
                # Convert all boxes at once
                x1 = (person_boxes[:, 1] * frame_w).astype(int)
                y1 = (person_boxes[:, 0] * frame_h).astype(int)
                x2 = (person_boxes[:, 3] * frame_w).astype(int)
                y2 = (person_boxes[:, 2] * frame_h).astype(int)
                
                # Create detections list
                detections = list(zip(x1, y1, x2, y2))
                count = len(detections)
                
                # Check for anomalies
                if count > 10:
                    anomalies = detections[10:]
            
            # Cache results
            self.last_detections = detections
            self.last_count = count
            
            # Update statistics
            processing_time = round((time.time() - start_time) * 1000, 2)
            with self.frame_lock:
                self.people_counts.append(count)
                self.anomalies_log.append(len(anomalies))
                self.processing_times.append(processing_time)
            
            # Draw results and cache processed frame
            processed_frame = self._draw_results(frame, detections, count)
            self.last_processed_frame = processed_frame[0].copy()
            
            # Send data to server if enough time has passed
            current_time = time.time()
            if current_time - self.last_send_time >= self.advanced_analytics_interval:
                self.last_send_time = current_time
                self._send_data_to_server(processed_frame[0], count, detections, anomalies)
            
            return processed_frame

        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            return frame, 0, [], []

    def _draw_cached_results(self, frame):
        """Draw cached detection results"""
        return self._draw_results(frame, self.last_detections, self.last_count)

    def _draw_results(self, frame, detections, count):
        """Draw detection results efficiently"""
        try:
            # Draw bounding boxes
            for (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add count text
            cv2.putText(frame, f"People: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return frame, count, detections, []
        except Exception as e:
            self.logger.error(f"Drawing results failed: {str(e)}")
            return frame, count, detections, []

    def get_statistics(self):
        """Get current statistics"""
        with self.frame_lock:
            return {
                "current_count": self.last_count,
                "average_count": np.mean(list(self.people_counts)) if self.people_counts else 0,
                "anomalies": len(self.anomalies_log),
                "fps": len(self.processing_times) / (sum(self.processing_times) / 1000) if self.processing_times else 0,
                "processing_time_ms": np.mean(list(self.processing_times)) if self.processing_times else 0
            }

    def _send_data_to_server(self, frame, count, detections, anomalies):
        """Send data to the server in a separate thread"""
        try:
            # Prepare data
            timestamp = datetime.now().isoformat()
            
            # Compress frame data
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Convert detections to list of lists (convert numpy types to Python native types)
            detections_list = []
            for det in detections:
                detections_list.append([int(x) for x in det])  # Convert numpy.int64 to Python int
            
            # Prepare payload with converted types
            payload = {
                "timestamp": timestamp,
                "count": int(count),  # Convert numpy.int64 to Python int
                "detections": detections_list,
                "frame_data": frame_data,
                "anomalies": int(len(anomalies)),  # Convert numpy.int64 to Python int
                "device_id": "raspberry_pi" if self.is_pi else "unknown"
            }
            
            # Send data in a separate thread
            if self.send_thread is None or not self.send_thread.is_alive():
                self.send_thread = Thread(target=self._send_data_thread, args=(payload,))
                self.send_thread.start()
                
        except Exception as e:
            self.logger.error(f"Error preparing data for server: {str(e)}")

    def _send_data_thread(self, payload):
        """Thread function to send data to server"""
        try:
            with self.send_lock:
                # Log the payload being sent
                self.logger.info(f"Sending data to server: {json.dumps(payload, indent=2)}")
                
                response = requests.post(
                    self.server_url,
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.logger.info("Data sent successfully to server")
                else:
                    self.logger.warning(f"Server returned status code: {response.status_code}")
                    # Log the response content for debugging
                    try:
                        error_content = response.json()
                        self.logger.error(f"Server error response: {error_content}")
                    except:
                        self.logger.error(f"Server error response: {response.text}")
                    
        except Exception as e:
            self.logger.error(f"Error sending data to server: {str(e)}")
            self.logger.error(f"Full error details: {traceback.format_exc()}")

    def __del__(self):
        """Cleanup"""
        pass  # Removed unnecessary cleanup for better performance