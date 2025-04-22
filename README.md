# 🚀 Crowd Analysis using Edge Computing

---

## 🧠 Overview

This project implements a **real-time crowd analysis system** using **edge computing**.  
It leverages **TensorFlow Lite** and **OpenCV** on lightweight devices (e.g., Raspberry Pi) to detect, count, and monitor crowds with **low latency** and **privacy preservation**.  
A **Tkinter GUI** and a **live dashboard** provide real-time visualizations of crowd analytics.

---

## 🌟 Features

- Real-time person detection using **SSD MobileNet** (TensorFlow Lite).
- Supports multiple video sources: **RTSP cameras**, **webcams**, and **local video files**.
- Lightweight **Tkinter GUI** for local monitoring.
- Edge-first processing ensures **data privacy** and **offline capability**.
- Metadata storage for trend analysis and event logging.
- Designed to run on **low-cost devices** (like Raspberry Pi 4).

---

## 🛠 Tech Stack

- **Frontend:** Tkinter (Python)
- **Backend (optional):** Flask (Python)
- **Machine Learning:** TensorFlow Lite (SSD MobileNet), OpenCV
- **Database:** CSV logging (MongoDB planned)
- **Protocols:** RTSP, HTTP/HTTPS

---

## 📁 Project Structure

```plaintext
├── analysis_results/       # Stores processed crowd data and logs
├── models/                 # TensorFlow Lite models
├── utils/                  # Helper utilities
├── crowd_analysis.py       # Main crowd detection logic
├── gui.py                  # Tkinter GUI for local monitoring
├── main.py                 # Entry point to launch application
├── requirements.txt        # Python libraries to install
```

---

## ⚙️ How to Setup and Run

### 1. Clone the Repository
```bash
git clone https://github.com/BabaAayam/Crowd-Github.git
cd Crowd-Github
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

### 4. How to Use
- After running `main.py`, a **Tkinter GUI** will open.
- Select the video input: **Webcam**, **RTSP Camera**, or **Local Video File**.
- The system will start real-time crowd detection.
- The GUI shows:
  - Crowd count
  - Live bounding boxes
  - System performance stats (FPS, CPU usage).

---

## 🎯 Use Cases

- **Public Safety:** Monitor crowd density at malls, concerts, metro stations.
- **Smart Cities:** Optimize urban infrastructure based on pedestrian flow.
- **Retail Analytics:** Analyze customer movements and optimize store layouts.
- **Emergency Response:** Early detection of crowding or stampedes.
- **Industrial Safety:** Monitor factory floors and ensure safe worker densities.

---

## 🔮 Future Scope

- Integration with **YOLOv8** for enhanced backend detection.
- **Real-time web dashboard** with crowd heatmaps and alerts.
- **MongoDB** database support for scalable metadata storage.
- **Predictive analytics** using time-series modeling.

---

## 👥 Contributors

- [Aayam Bajaj](https://github.com/BabaAayam)
- Atharva Dhokte
- Yuvrajsing Bahure

---

# 🎉 Thank You for Visiting this Project!
