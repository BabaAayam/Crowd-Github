import time
import cv2
import numpy as np
import tensorflow as tf
import requests
import json

class CrowdAnalyzer:
    def __init__(self, model_path="models/ssd_mobilenet.tflite", server_url="http://localhost:5000/receive_data",edge_mode=False):
        # Model initialization
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        
        # Server configuration
        self.server_url = server_url
        self.frame_counter = 0
        self.last_send_time = time.time()
        
        # Data tracking
        self.people_counts = []
        self.anomalies_log = []
        self.processing_times = []

    def process_frame(self, frame):
        try:
            # Frame skipping logic (process every 3rd frame)
            self.frame_counter += 1
            if self.frame_counter % 3 != 0:
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

            count = 0
            detections = []
            anomalies = []
            h, w = frame.shape[:2]

            for i in range(num_detections):
                if scores[i] > 0.5 and classes[i] == 0:
                    ymin, xmin, ymax, xmax = boxes[i]
                    x1, y1 = int(xmin * w), int(ymin * h)
                    x2, y2 = int(xmax * w), int(ymax * h)
                    detections.append((x1, y1, x2, y2))
                    count += 1
                    if count > 10:  # Anomaly threshold
                        anomalies.append((x1, y1, x2, y2))

            

            # Store metrics
            processing_time = round((time.time() - start_time) * 1000, 2)
            self.people_counts.append(count)
            self.anomalies_log.append(len(anomalies))
            self.processing_times.append(processing_time)

            # Conditional HTTP send
            current_time = time.time()
            if count > 10 or (current_time - self.last_send_time) > 30:  # 30s heartbeat
                self.send_data_to_server(count, detections)
                self.last_send_time = current_time

            return frame, count, detections, anomalies

        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
            return frame, 0, [], []

    def send_data_to_server(self, count, detections):
        """Optimized HTTP POST with frame skipping"""
        try:
            payload = {
                "count": count,
                "detections": detections,
                "timestamp": time.time()
            }
            response = requests.post(
                self.server_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=2  # 2-second timeout
            )
            if response.status_code != 200:
                print(f"[WARNING] Server response: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] HTTP send failed: {str(e)}")

# import cv2
# import numpy as np
# import tensorflow.lite as tflite

# # Load the TFLite model
# interpreter = tflite.Interpreter(model_path="models/ssd_mobilenet.tflite")
# interpreter.allocate_tensors()

# # Get model details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# height = input_details[0]['shape'][1]
# width = input_details[0]['shape'][2]

# # Open webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the image
#     resized_frame = cv2.resize(frame, (width, height))
#     input_data = np.expand_dims(resized_frame, axis=0)

#     # Run inference
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     # Extract results
#     boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
#     classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
#     scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence Scores

#     # Draw detections
#     for i in range(len(scores)):
#      if scores[i] > 0.5:  # Confidence threshold
#         y_min, x_min, y_max, x_max = boxes[i]  # Normalized coordinates

#         # Convert to pixel values
#         x_min = int(x_min * frame.shape[1])
#         y_min = int(y_min * frame.shape[0])
#         x_max = int(x_max * frame.shape[1])
#         y_max = int(y_max * frame.shape[0])

#         # Draw bounding box
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         # Display label (assuming class ID 0 = "Person")
#         label = "Person" if int(classes[i]) == 0 else "Unknown"
#         cv2.putText(frame, label, (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#     # Show output
#     cv2.imshow("Crowd Analysis", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
