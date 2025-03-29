import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import time
import numpy as np
from crowd_analysis import CrowdAnalyzer  # Import your CrowdAnalyzer class

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CrowdAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Analysis")
        self.root.geometry("1200x800")  # Larger window

        # Variables
        self.source_type = tk.StringVar(value="video")  # Default: Video file
        self.rtsp_url = tk.StringVar()
        self.file_path = tk.StringVar()
        self.running = False
        self.cap = None  # Video capture object
        self.analyzer = CrowdAnalyzer()  # Load the analysis model

        # GUI Setup
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI elements including buttons, labels, and video frame."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Source selection
        ttk.Label(control_frame, text="Source:").grid(row=0, column=0)
        ttk.Radiobutton(control_frame, text="Video", variable=self.source_type, value="video").grid(row=0, column=1)
        # Add "Show Results" button next to Browse
        ttk.Button(control_frame, text="Show Results", command=self.show_results).grid(row=2, column=3)

                        
        # RTSP URL input
        ttk.Label(control_frame, text="RTSP URL:").grid(row=1, column=0)
        ttk.Entry(control_frame, textvariable=self.rtsp_url, width=40).grid(row=1, column=1, columnspan=2)
        
        # File browse for videos
        ttk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=2, column=0)
        ttk.Entry(control_frame, textvariable=self.file_path, width=40).grid(row=2, column=1, columnspan=2)
        
        # Start/Stop buttons
        ttk.Button(control_frame, text="Start", command=self.start_analysis).grid(row=3, column=0, pady=10)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=3, column=1)

        # Video display (center)
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # People counter overlay (top-right)
        self.count_label = ttk.Label(
            self.video_frame,
            text="People: 0",
            font=('Arial', 24, 'bold'),
            foreground="red",
            background="black"
        )
        self.count_label.place(relx=0.95, rely=0.05, anchor=tk.NE)  # Overlay on video

        # Results Section (bottom)
        self.results_label = ttk.Label(self.root, text="Results: -", font=("Arial", 12), foreground="blue")

        self.results_label.pack(pady=10)

    def browse_file(self):
        """Open file dialog to select a video file."""
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if filename:
            self.file_path.set(filename)  # Set selected file path

    def start_analysis(self):
        """Start processing video from file or RTSP."""
        if self.source_type.get() == "cctv":
            source = self.rtsp_url.get()
            if not source:
                print("[ERROR] No RTSP URL provided!")
                return
        else:
            source = self.file_path.get()
            if not source:
                print("[ERROR] No video file selected!")
                return

        # Open video stream
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open source: {source}")
            return
        
        self.running = True
        self.update_frame()

    def update_frame(self):
        """Read and process the next frame."""
        if not self.running:
            return

        start_time = time.time()  # Start processing time

        ret, frame = self.cap.read()
        if not ret:
            print("[INFO] End of video or stream error.")
            self.stop()
            return

        # Process frame
        processed_frame, count, boxes, anomalies = self.analyzer.process_frame(frame)

        # ✅ Ensure self.analyzer.people_counts exists
        if not hasattr(self.analyzer, "people_counts"):
            self.analyzer.people_counts = []
            self.analyzer.anomalies_log = []
            self.analyzer.processing_times = []

        # ✅ Append current data to lists
        self.analyzer.people_counts.append(count)
        self.analyzer.anomalies_log.append(len(anomalies))
        self.analyzer.processing_times.append(round((time.time() - start_time) * 1000, 2))

        # Draw bounding boxes
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Update count label
        self.count_label.config(text=f"People: {count}")

        # Resize frame to fit window
        h, w = processed_frame.shape[:2]
        max_width = self.video_frame.winfo_width()
        max_height = self.video_frame.winfo_height()

        if w > max_width or h > max_height:
            scale = min(max_width/w, max_height/h)
            processed_frame = cv2.resize(processed_frame, (int(w*scale), int(h*scale)))

        # Convert frame to ImageTk format
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Results Section
        processing_time = round((time.time() - start_time) * 1000, 2)
        anomaly_text = "✔ No anomalies detected" if not anomalies else "⚠ Anomalies detected!"

        results_text = (
            f"🔹 Detected People: {count}  |  "
            f"🔹 Bounding Boxes: {len(boxes)}  |  "
            f"🔹 Resolution: {w}x{h}  |  "
            f"🔹 Processing Time: {processing_time} ms  |  "
            f"{anomaly_text}"
        )

        self.results_label.config(text=results_text)

        # Next frame
        self.root.after(30, self.update_frame)

    def stop(self):
        """Stop video processing."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")

   
    def show_results(self):
        """Create a new window to display analysis results."""
        results_window = tk.Toplevel(self.root)
        results_window.title("Analysis Results")
        results_window.geometry("700x550")

        ttk.Label(results_window, text="Crowd Analysis Summary", font=("Arial", 14, "bold")).pack(pady=10)

        # ✅ Ensure there is data available
        if not hasattr(self.analyzer, "people_counts") or not self.analyzer.people_counts:
            ttk.Label(results_window, text="No data available. Run analysis first!", foreground="red").pack(pady=20)
            return

        # ✅ Calculate total people detected across all frames
        total_people_detected = sum(self.analyzer.people_counts)

        # ✅ Display total people detected
        ttk.Label(results_window, text=f"📊 Total People Detected: {total_people_detected}", 
                font=("Arial", 12, "bold"), foreground="blue").pack(pady=5)

        # ✅ Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(6, 8))

        # 1️⃣ Plot People Count Over Time
        axs[0].plot(self.analyzer.people_counts, label="People Count", color="blue", marker="o")
        axs[0].set_title("People Count Over Time")
        axs[0].set_xlabel("Frame Number")
        axs[0].set_ylabel("People Count")
        axs[0].legend()
        axs[0].grid()

        # 2️⃣ Plot Anomalies Over Time
        axs[1].plot(self.analyzer.anomalies_log, label="Anomalies", color="red", marker="s")
        axs[1].set_title("Anomalies Over Time")
        axs[1].set_xlabel("Frame Number")
        axs[1].set_ylabel("Anomaly Count")
        axs[1].legend()
        axs[1].grid()

        # 3️⃣ Plot Processing Time Per Frame
        axs[2].plot(self.analyzer.processing_times, label="Processing Time", color="green", marker="^")
        axs[2].set_title("Frame Processing Time")
        axs[2].set_xlabel("Frame Number")
        axs[2].set_ylabel("Time (ms)")
        axs[2].legend()
        axs[2].grid()

        # Embed Matplotlib graph in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack()



# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CrowdAnalysisApp(root)
#     root.mainloop()
