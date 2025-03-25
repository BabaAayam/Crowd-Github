import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
from crowd_analysis import CrowdAnalyzer

class CrowdAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Analysis")
        self.root.geometry("1200x800")  # Larger window
        
        # Variables
        self.source_type = tk.StringVar(value="video")
        self.rtsp_url = tk.StringVar()
        self.file_path = tk.StringVar()
        self.running = False

        # GUI Setup
        self.setup_ui()

    def setup_ui(self):
        # Source selection (top-left)
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Source:").grid(row=0, column=0)
        ttk.Radiobutton(control_frame, text="Video", variable=self.source_type, value="video").grid(row=0, column=1)
        ttk.Radiobutton(control_frame, text="CCTV (RTSP)", variable=self.source_type, value="cctv").grid(row=0, column=2)
        
        ttk.Label(control_frame, text="RTSP URL:").grid(row=1, column=0)
        ttk.Entry(control_frame, textvariable=self.rtsp_url, width=40).grid(row=1, column=1, columnspan=2)
        
        ttk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=2, column=0)
        ttk.Entry(control_frame, textvariable=self.file_path, width=40).grid(row=2, column=1, columnspan=2)
        
        ttk.Button(control_frame, text="Start", command=self.start_analysis).grid(row=3, column=0, pady=10)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=3, column=1)

        # Video display (center)
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # People counter (top-right overlay)
        self.count_label = ttk.Label(
            self.video_frame,
            text="People: 0",
            font=('Arial', 24, 'bold'),
            foreground="red",
            background="black"
        )
        self.count_label.place(relx=0.95, rely=0.05, anchor=tk.NE)  # Overlay on video

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if filename:
            self.file_path.set(filename)

    def start_analysis(self):
        if self.running:
            return

        # Get source
        if self.source_type.get() == "video":
            source = self.file_path.get()
            self.cap = cv2.VideoCapture(source)
        elif self.source_type.get() == "cctv":
            source = self.rtsp_url.get()
            
            # RTSP Fix for Windows
            if not source.startswith("rtsp://"):
                print("Error: RTSP URL must start with 'rtsp://'")
                return
                
            # Add TCP transport if not already specified
            if "?" not in source:
                source += "?tcp"
                
            # Set larger buffer for Windows
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:  # camera
            source = 0
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open video source: {source}")
            if self.source_type.get() == "cctv":
                print("\nRTSP Troubleshooting:")
                print("1. Verify URL format: rtsp://username:password@ip:port/stream")
                print("2. Try adding '?tcp' to force TCP mode")
                print("3. Test the stream in VLC media player first")
            return

        self.analyzer = CrowdAnalyzer()
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        # Process frame
        processed_frame, count, boxes = self.analyzer.process_frame(frame)

        # Draw boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Update counter
        self.count_label.config(text=f"People: {count}")

        # Resize frame to fit window
        h, w = processed_frame.shape[:2]
        max_width = self.video_frame.winfo_width()
        max_height = self.video_frame.winfo_height()
        
        if w > max_width or h > max_height:
            scale = min(max_width/w, max_height/h)
            processed_frame = cv2.resize(processed_frame, (int(w*scale), int(h*scale)))

        # Display
        img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Next frame
        self.root.after(30, self.update_frame)