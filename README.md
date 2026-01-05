# 🚶 Live People Counter

<p align="center">
  <a href="https://www.python.org/" target="_blank"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/></a>
  <a href="https://github.com/ultralytics/ultralytics" target="_blank"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg" alt="YOLOv8"/></a>
  <a href="https://opencv.org/" target="_blank"><img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV"/></a>
  <a href="LICENSE" title="MIT License - Open Source"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/></a>
</p>

<p align="center">
  <strong>Real-time people counting system using YOLOv8 and OpenCV</strong>
</p>

<p align="center">
  <a href="#features"><strong>Features</strong></a> •
  <a href="#installation"><strong>Installation</strong></a> •
  <a href="#usage"><strong>Usage</strong></a> •
  <a href="#how-it-works"><strong>How It Works</strong></a> •
  <a href="#configuration"><strong>Configuration</strong></a>
</p>

---

## <a id="features"></a>✨ Features

- 🎯 **Real-time Detection** - Uses YOLOv8 for accurate person detection
- 📊 **Bi-directional Counting** - Tracks people entering and exiting
- 🔢 **Live Statistics** - Shows IN, OUT, and Total Inside counts
- 📝 **Activity Logging** - Records all movements with timestamps
- 🖥️ **Live Video Feed** - Visualizes detection with bounding boxes and tracking lines
- ⚡ **High Performance** - Optimized for real-time processing

## 🎬 Demo

<p align="center">
  <img src="demo/demo.gif" alt="Demo" width="600"/>
</p>

## 📋 Requirements

- Python 3.8+
- Webcam or video input device
- NVIDIA GPU (recommended for better performance)

## <a id="installation"></a>🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/live-people-counter.git
cd live-people-counter
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 model

The model will be downloaded automatically on first run, or you can download it manually:

```bash
# Using Python
from ultralytics import YOLO
model = YOLO('yolov8x.pt')  # Downloads automatically
```

## <a id="usage"></a>🚀 Usage

### Basic Usage

```bash
python PeopleCounter.py
```

### Controls

- Press **`Esc`** to exit the application
- The window is **resizable** - drag to adjust size

### Output

- **Live video** with detection boxes and counting lines
- **Log file** (`log.txt`) with detailed movement records

## <a id="how-it-works"></a>🔧 How It Works

### Detection Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Capture    │───▶│   YOLOv8    │───▶│  Tracking   │───▶│  Counting   │
│   Frame     │    │  Detection  │    │   Logic     │    │   Logic     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Counting Logic

The system uses **two virtual lines** to determine movement direction:

- **Blue Line (Line Down)** - Lower boundary
- **Red Line (Line Up)** - Upper boundary

| Direction | Description |
|-----------|-------------|
| **Going UP** | Person crosses from below Line Down to above Line Up → **EXIT** |
| **Going DOWN** | Person crosses from above Line Up to below Line Down → **ENTER** |

### Zone Layout

```
     ┌───────────────────────────────────┐
     │         Up Limit (white)          │  ← Tracking boundary
     │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
     │         Line Up (red)             │  ← Upper counting line
     │                                   │
     │         COUNTING ZONE             │
     │                                   │
     │         Line Down (blue)          │  ← Lower counting line
     │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
     │         Down Limit (white)        │  ← Tracking boundary
     └───────────────────────────────────┘
```

## <a id="configuration"></a>⚙️ Configuration

You can modify these parameters in `PeopleCounter.py`:

### Detection Settings

```python
model = YOLO('yolov8x.pt')  # Model: yolov8n.pt (fast) to yolov8x.pt (accurate)
results = model(frame, conf=0.5)  # Confidence threshold (0.0 - 1.0)
```

### Line Positions

```python
line_up = int(2 * (h / 5))    # Upper line position
line_down = int(3 * (h / 5))  # Lower line position
up_limit = int(1 * (h / 5))   # Upper tracking boundary
down_limit = int(4 * (h / 5)) # Lower tracking boundary
```

### Video Source

```python
cap = cv.VideoCapture(0)  # 0 = default webcam
# cap = cv.VideoCapture('video.mp4')  # Or use a video file
# cap = cv.VideoCapture('rtsp://...')  # Or RTSP stream
```

### Model Comparison

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| `yolov8n.pt` | ⚡⚡⚡ Fastest | Good | 6 MB |
| `yolov8s.pt` | ⚡⚡ Fast | Better | 22 MB |
| `yolov8m.pt` | ⚡ Medium | Great | 50 MB |
| `yolov8x.pt` | Slower | Best | 131 MB |

## 📁 Project Structure

```
live-people-counter/
├── PeopleCounter.py   # Main application
├── Person.py          # Person tracking class
├── coco.txt           # COCO class labels
├── yolov8x.pt         # YOLOv8 model (download separately)
├── log.txt            # Activity log (generated)
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── LICENSE            # MIT License
```

## 📊 Log Format

The `log.txt` file records all movements:

```
ID: 1 crossed going down at Sun Jan 04 10:30:45 2026. Total inside: 1
ID: 2 crossed going down at Sun Jan 04 10:31:02 2026. Total inside: 2
ID: 1 crossed going up at Sun Jan 04 10:35:20 2026. Total inside: 1
```

## 🐛 Troubleshooting

### Camera not opening

```python
# Check available cameras
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
```

### Low FPS

- Use a lighter model: `yolov8n.pt` or `yolov8s.pt`
- Reduce frame resolution
- Use GPU acceleration (CUDA)

### CUDA/GPU Support

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the <a href="LICENSE" title="MIT License - Open Source">LICENSE</a> file for details.

## 🙏 Acknowledgments

- <a href="https://github.com/ultralytics/ultralytics" target="_blank">Ultralytics YOLOv8</a> - State-of-the-art object detection
- <a href="https://opencv.org/" target="_blank">OpenCV</a> - Computer vision library

## 👨‍💻 Author

**Subodh Adhikari**

- GitHub: <a href="https://github.com/adhikarisubodh9991" target="_blank">@adhikarisubodh9991</a>

---

<p align="center">
  ⭐ Star this repo if you find it useful!
</p>
