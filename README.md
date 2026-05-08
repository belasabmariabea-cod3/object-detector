# 🎥 AI Vision Dashboard (Live Object Detection)

A real-time AI-powered object detection web application built using **Streamlit**, **YOLOv8 (Ultralytics)**, OpenCV, and WebRTC.  
It detects and tracks objects from a live camera feed with a modern pastel-themed UI.

---

## 🚀 Features

- 🎯 Real-time object detection using YOLOv8 (`yolov8n.pt`)
- 📷 Live webcam streaming via `streamlit-webrtc`
- 📊 Object counting dashboard (HUD overlay)
- 🎨 Modern pastel UI with custom CSS styling
- 🧠 AI-powered detection with confidence scoring
- 🔄 Start / Stop camera control buttons
- ⚡ Lightweight and fast processing

---

## 🛠️ Tech Stack

- **Python 3.8+**
- [Streamlit](https://streamlit.io/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- [Av (PyAV)](https://github.com/PyAV-Org/PyAV)

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-vision-dashboard.git
cd ai-vision-dashboard
2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3. Install dependencies
pip install streamlit
pip install streamlit-webrtc
pip install ultralytics
pip install opencv-python
pip install av

▶️ Run the App
streamlit run app.py
📸 How It Works

Click 🟢 TURN ON CAMERA
Your webcam feed starts streaming
YOLOv8 detects objects in real time
Bounding boxes + labels + confidence appear
HUD shows detected object counts

Click 🔴 TURN OFF CAMERA to stop
🧠 Model Information
Model used: yolov8n.pt (Nano version of YOLOv8)
Trained on COCO dataset (80 common object classes)
Optimized for real-time performance

🎨 UI Design
Pastel gradient background
Glassmorphism camera container
Smooth typography using Poppins font
Minimalist HUD overlay with transparency
Responsive centered layout

⚙️ Configuration Notes
Default camera resolution: 640x480
Confidence threshold: 0.5
Audio is disabled for performance
Uses Google STUN server for WebRTC connection

📌 Future Improvements
🔊 Sound alerts when objects are detected
📈 Live analytics dashboard (charts)
🧍 Custom object training support
📱 Mobile optimization
💾 Detection recording/export feature

🧑‍💻 Author
Developer: Maria Bea Belasa
Developer for educatinal purposes in Computer Vison & Ai Learning

Feel free to modify and enhance this project 🚀

📄 License
This project is open-source and free to use under the MIT License.