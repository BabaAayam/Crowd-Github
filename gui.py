import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import time
import platform
from crowd_analysis import CrowdAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import requests
from datetime import datetime

class CrowdAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Analysis")
        
        # Device detection
        self.is_edge_device = self.detect_edge_device()
        self.configure_window_size()
        
        # Variables
        self.source_type = tk.StringVar(value="video")
        self.rtsp_url = tk.StringVar()
        self.file_path = tk.StringVar()
        self.running = False
        self.cap = None
        
        # Initialize analyzer with empty metrics lists
        self.analyzer = CrowdAnalyzer(
            server_url="http://localhost:5000/receive_data",
            edge_mode=self.is_edge_device
        )
        self.analyzer.processing_times = []
        self.analyzer.people_counts = []
        self.analyzer.anomalies_log = []
        
        # GUI Setup
        self.setup_ui()
        self.frame_counter = 0
        self.last_frame_time = time.time()
        self.start_connection_checker()

    def detect_edge_device(self):
        """Check if running on Raspberry Pi"""
        if platform.system() == 'Linux':
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    return 'raspberry pi' in f.read().lower()
            except:
                return platform.machine() in ('armv7l', 'aarch64')
        return False

    def start_connection_checker(self):
        """Background thread to check server connection"""
        def checker():
            while True:
                try:
                    response = requests.get(
                        self.analyzer.server_url.replace("receive_data", "ping"),
                        timeout=2
                    )
                    status = "🟢 Connected" if response.status_code == 200 else "🔴 Server Error"
                    color = "green" if response.status_code == 200 else "red"
                except:
                    status = "🔴 Disconnected"
                    color = "red"
                
                self.connection_label.config(text=status, foreground=color)
                time.sleep(5)
        
        threading.Thread(target=checker, daemon=True).start()

    def configure_window_size(self):
        """Set window size based on device type"""
        if self.is_edge_device:
            self.root.geometry("800x480")  # Raspberry Pi optimized size
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry("1200x800")

    def setup_ui(self):
        """Setup the UI elements"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Device status
        device_status = "🟢 Edge Mode" if self.is_edge_device else "🔴 Desktop Mode"
        ttk.Label(control_frame, text=device_status, 
                 foreground="green" if self.is_edge_device else "red").grid(row=0, column=4, padx=10)
        
        # Connection status
        self.connection_label = ttk.Label(control_frame, text="⚪ Checking...", foreground="gray")
        self.connection_label.grid(row=0, column=5, padx=10)
        
        # Source selection
        ttk.Label(control_frame, text="Source:").grid(row=0, column=0)
        ttk.Radiobutton(control_frame, text="Video", variable=self.source_type, value="video").grid(row=0, column=1)
        ttk.Radiobutton(control_frame, text="RTSP", variable=self.source_type, value="rtsp").grid(row=0, column=2)
        
        # RTSP URL input
        ttk.Label(control_frame, text="RTSP URL:").grid(row=1, column=0)
        ttk.Entry(control_frame, textvariable=self.rtsp_url, width=40).grid(row=1, column=1, columnspan=2)
        
        # File browse for videos
        ttk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=2, column=0)
        ttk.Entry(control_frame, textvariable=self.file_path, width=40).grid(row=2, column=1, columnspan=2)
        ttk.Button(control_frame, text="Show Results", command=self.show_results).grid(row=2, column=3)
        
        # Start/Stop buttons
        ttk.Button(control_frame, text="Start", command=self.start_analysis).grid(row=3, column=0, pady=10)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=3, column=1)

        # Video display
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # People counter overlay
        self.count_label = ttk.Label(
            self.video_frame,
            text="People: 0",
            font=('Arial', 24, 'bold'),
            foreground="red",
            background="black"
        )
        self.count_label.place(relx=0.95, rely=0.05, anchor=tk.NE)

        # Results Section
        self.results_label = ttk.Label(
            self.root, 
            text="🔹 Ready | FPS: - | Delay: - ms", 
            font=("Arial", 12), 
            foreground="blue"
        )
        self.results_label.pack(pady=10)

    def browse_file(self):
        """Open file dialog to select a video file."""
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if filename:
            self.file_path.set(filename)

    def start_analysis(self):
        """Start processing video from file or RTSP."""
        if self.source_type.get() == "rtsp":
            source = self.rtsp_url.get()
            if not source:
                print("[ERROR] No RTSP URL provided!")
                return
        else:
            source = self.file_path.get()
            if not source:
                print("[ERROR] No video file selected!")
                return

        # Open video stream with device-specific settings
        self.cap = cv2.VideoCapture(source)
        if self.is_edge_device:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS on Pi
        
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open source: {source}")
            return
        
        self.running = True
        self.frame_counter = 0
        self.last_frame_time = time.time()
        self.update_frame()

    def update_frame(self):
        """Read and process the next frame."""
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("[INFO] End of video or stream error.")
            self.stop()
            return

        # Process frame
        processed_frame, count, boxes, anomalies = self.analyzer.process_frame(frame)

        # Calculate FPS and delay
        current_time = time.time()
        fps = 1 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time

        # Draw bounding boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Update UI with safe metrics access
        self.count_label.config(text=f"People: {count}")
        
        # Safe delay text handling
        delay_text = "- ms"
        if hasattr(self.analyzer, 'processing_times') and self.analyzer.processing_times:
            delay_text = f"{self.analyzer.processing_times[-1]} ms"
        
        anomaly_text = "✔ Normal" if not anomalies else f"⚠ {len(anomalies)} Anomalies!"
        self.results_label.config(text=(
            f"🔹 People: {count} | FPS: {fps:.1f} | "
            f"Delay: {delay_text} | {anomaly_text}"
        ))

        # Resize for display
        h, w = processed_frame.shape[:2]
        max_width = self.video_frame.winfo_width()
        max_height = self.video_frame.winfo_height()
        if w > max_width or h > max_height:
            scale = min(max_width/w, max_height/h)
            processed_frame = cv2.resize(processed_frame, (int(w*scale), int(h*scale)))

        # Convert and display frame
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Schedule next frame
        self.root.after(30, self.update_frame)

    def stop(self):
        """Stop video processing."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.connection_label.config(text="⚪ Disconnected", foreground="gray")

    def show_results(self):
        """Display analysis results in new window."""
        results_window = tk.Toplevel(self.root)
        results_window.title("Analysis Results")
        results_window.geometry("700x550")

        ttk.Label(results_window, text="Crowd Analysis Summary", 
                 font=("Arial", 14, "bold")).pack(pady=10)

        if not hasattr(self.analyzer, "people_counts") or not self.analyzer.people_counts:
            ttk.Label(results_window, text="No data available. Run analysis first!", 
                     foreground="red").pack(pady=20)
            return

        total_people = sum(self.analyzer.people_counts)
        ttk.Label(results_window, 
                 text=f"📊 Total People Detected: {total_people}", 
                 font=("Arial", 12, "bold"), 
                 foreground="blue").pack(pady=5)

        # Create plots
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))
        
        # People count plot
        axs[0].plot(self.analyzer.people_counts, color="blue", marker="o")
        axs[0].set_title("People Count Over Time")
        axs[0].grid()
        
        # Anomalies plot
        axs[1].plot(self.analyzer.anomalies_log, color="red", marker="s")
        axs[1].set_title("Anomalies Over Time")
        axs[1].grid()
        
        # Processing time plot
        axs[2].plot(self.analyzer.processing_times, color="green", marker="^")
        axs[2].set_title("Processing Time (ms)")
        axs[2].grid()

        # Embed plots
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CrowdAnalysisApp(root)
#     root.mainloop()