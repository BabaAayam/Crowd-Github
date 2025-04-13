import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import time
import platform
import os
import requests
import threading
from datetime import datetime
from crowd_analysis import CrowdAnalyzer
import logging
from collections import deque

class CrowdAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.logger = logging.getLogger(__name__)
        
        # Initialize all required attributes
        self.cap = None
        self.running = False
        self.frame_counter = 0
        self.last_frame_time = time.time()
        self.source_type = tk.StringVar(value="camera")
        
        # Device detection
        self.is_edge_device = self.detect_edge_device()
        self.logger.info(f"Running on edge device: {self.is_edge_device}")
        
        # Server configuration
        self.server_url = tk.StringVar(value="http://192.168.191.1:5000/receive_data")  # Update with your Windows server IP
        self.server_status = tk.StringVar(value="Disconnected")
        self.server_status_color = "red"
        
        # Initialize analyzer
        self.analyzer = CrowdAnalyzer(
            edge_mode=self.is_edge_device,
            server_url=self.server_url.get()
        )
        
        # Frame buffering
        self.frame_buffer = deque(maxlen=3)  # Buffer for smoother display
        self.last_display_time = time.time()
        self.target_fps = 30  # Set to standard video FPS
        self.frame_interval = 1.0 / self.target_fps
        
        # UI update optimization
        self.last_stats_update = 0
        self.stats_update_interval = 0.5  # Update stats every 0.5 seconds
        self.last_fps_update = 0
        self.fps_update_interval = 0.2  # Update FPS every 0.2 seconds
        
        # Setup GUI
        self.setup_ui()
        
        # Start connection checker
        self.start_connection_checker()

    def detect_edge_device(self):
        """Check if running on Raspberry Pi"""
        try:
            if platform.system() == 'Linux':
                with open('/proc/device-tree/model', 'r') as f:
                    return 'raspberry pi' in f.read().lower()
        except:
            return platform.machine() in ('armv7l', 'aarch64')
        return False

    def setup_ui(self):
        """Complete GUI setup with modern features"""
        # Configure style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Source selection
        source_frame = ttk.Frame(control_frame)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(source_frame, text="Source:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Camera", variable=self.source_type, 
                       value="camera", command=self.on_source_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Video File", variable=self.source_type, 
                       value="file", command=self.on_source_change).pack(side=tk.LEFT, padx=5)
        
        # File browser
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Create statistics labels
        self.stats_vars = {
            "People Count": tk.StringVar(value="0"),
            "Processing Time": tk.StringVar(value="0 ms"),
            "FPS": tk.StringVar(value="0"),
            "Anomalies": tk.StringVar(value="0")
        }
        
        for i, (label, var) in enumerate(self.stats_vars.items()):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=0, column=i*2, padx=5, pady=5)
            ttk.Label(stats_frame, textvariable=var).grid(row=0, column=i*2+1, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
        
        # Add server configuration panel
        server_frame = ttk.LabelFrame(main_frame, text="Server Configuration", padding=10)
        server_frame.pack(fill=tk.X, pady=5)
        
        # Server URL entry
        url_frame = ttk.Frame(server_frame)
        url_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(url_frame, text="Server URL:").pack(side=tk.LEFT, padx=5)
        url_entry = ttk.Entry(url_frame, textvariable=self.server_url, width=50)
        url_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Server status
        status_frame = ttk.Frame(server_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(status_frame, textvariable=self.server_status, 
                                    foreground=self.server_status_color)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Update server button
        update_button = ttk.Button(server_frame, text="Update Server", 
                                 command=self.update_server_config)
        update_button.pack(pady=5)
        
        # Configure for Pi's display
        if self.is_edge_device:
            self.root.attributes('-fullscreen', True)
            # Add exit button for fullscreen mode
            exit_button = ttk.Button(control_frame, text="Exit", command=self.root.quit)
            exit_button.pack(side=tk.RIGHT, padx=5)

    def on_source_change(self):
        """Handle source type change"""
        if self.source_type.get() == "camera":
            self.file_entry.config(state=tk.DISABLED)
            self.browse_button.config(state=tk.DISABLED)
        else:
            self.file_entry.config(state=tk.NORMAL)
            self.browse_button.config(state=tk.NORMAL)

    def browse_file(self):
        """File browser implementation"""
        filename = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f"Selected file: {os.path.basename(filename)}")

    def start_analysis(self):
        """Start video analysis"""
        try:
            # Update analyzer with current server URL
            self.analyzer.server_url = self.server_url.get()
            
            if self.source_type.get() == "camera":
                self.cap = cv2.VideoCapture(0)
            else:
                if not self.file_path.get():
                    self.status_var.set("Please select a video file")
                    return
                self.cap = cv2.VideoCapture(self.file_path.get())
            
            if not self.cap.isOpened():
                self.status_var.set("Error: Could not open video source")
                return
            
            # Configure video capture
            if self.is_edge_device:
                # Optimize for Pi's camera
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set to standard FPS
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Try to use hardware acceleration
                try:
                    self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except:
                    self.logger.warning("Hardware acceleration not available")
            
            # Initialize timing variables
            self.last_display_time = time.time()
            self.last_stats_update = 0
            self.last_fps_update = 0
            self.last_frame_time = time.time()
            
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Analysis running...")
            
            # Start frame update loop
            self.update_frame()
            
        except Exception as e:
            self.logger.error(f"Error starting analysis: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.stop_capture()

    def update_frame(self):
        """Update video frame with analysis results"""
        if not self.running or not hasattr(self, 'cap'):
            return
        
        try:
            current_time = time.time()
            
            # Read frame without clearing buffer to maintain normal speed
            ret, frame = self.cap.read()
            
            if not ret:
                self.stop_capture()
                return
            
            # Process frame
            result = self.analyzer.process_frame(frame)
            if result:
                processed_frame, count, detections, anomalies = result
                
                # Add to frame buffer
                self.frame_buffer.append((processed_frame, count, detections, anomalies))
                
                # Update display if enough time has passed
                if current_time - self.last_display_time >= self.frame_interval:
                    self.last_display_time = current_time
                    
                    # Get latest frame from buffer
                    if self.frame_buffer:
                        processed_frame, count, detections, anomalies = self.frame_buffer[-1]
                        
                        # Convert and display frame efficiently
                        if processed_frame is not None:
                            # Resize frame to fit display
                            h, w = processed_frame.shape[:2]
                            display_w = min(640, w)
                            display_h = int(h * (display_w / w))
                            
                            if display_w != w:
                                processed_frame = cv2.resize(processed_frame, (display_w, display_h),
                                                           interpolation=cv2.INTER_AREA)
                            
                            # Convert to RGB and create image
                            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(img)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.video_label.imgtk = imgtk
                            self.video_label.config(image=imgtk)
                    
                    # Update statistics less frequently
                    if current_time - self.last_stats_update >= self.stats_update_interval:
                        self.last_stats_update = current_time
                        stats = self.analyzer.get_statistics()
                        self.stats_vars["People Count"].set(str(count))
                        self.stats_vars["Anomalies"].set(str(stats['anomalies']))
                        self.stats_vars["Processing Time"].set(f"{stats['processing_time_ms']:.1f} ms")
                    
                    # Update FPS more frequently
                    if current_time - self.last_fps_update >= self.fps_update_interval:
                        self.last_fps_update = current_time
                        fps = 1.0 / (current_time - self.last_frame_time)
                        self.last_frame_time = current_time
                        self.stats_vars["FPS"].set(f"{fps:.1f}")
            
            # Schedule next update with minimal delay
            self.root.after(1, self.update_frame)
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            self.status_var.set(f"Error processing frame: {str(e)}")
            self.stop_capture()

    def stop_capture(self):
        """Stop video capture and analysis"""
        self.running = False
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Analysis stopped")

    def update_server_config(self):
        """Update server configuration"""
        try:
            # Ensure URL ends with /receive_data
            server_url = self.server_url.get().strip()
            if not server_url.endswith('/receive_data'):
                if server_url.endswith('/'):
                    server_url += 'receive_data'
                else:
                    server_url += '/receive_data'
            
            # Update analyzer with new server URL
            self.analyzer.server_url = server_url
            self.server_url.set(server_url)  # Update the displayed URL
            self.status_var.set("Server configuration updated")
            self.logger.info(f"Server URL updated to: {server_url}")
        except Exception as e:
            self.status_var.set(f"Error updating server: {str(e)}")
            self.logger.error(f"Error updating server: {str(e)}")

    def start_connection_checker(self):
        """Check server connection status"""
        def check_connection():
            try:
                # Check base URL without /receive_data
                base_url = self.server_url.get().replace('/receive_data', '')
                response = requests.get(base_url, timeout=2)
                if response.status_code == 200:
                    self.server_status.set("Connected")
                    self.server_status_color = "green"
                else:
                    self.server_status.set("Connection Error")
                    self.server_status_color = "red"
            except:
                self.server_status.set("Disconnected")
                self.server_status_color = "red"
            
            # Update status label color
            self.status_label.config(foreground=self.server_status_color)
            
            # Schedule next check
            self.root.after(5000, check_connection)
        
        check_connection()

    def __del__(self):
        """Cleanup"""
        self.stop_capture()