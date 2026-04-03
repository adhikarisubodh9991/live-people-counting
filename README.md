# 🚶 Live People Counter

A smart, interactive people counting system using YOLOv11 (with optional custom B4rtekk1 backend) and a user-friendly setup wizard.

## ✨ Key Features

- **Accurate Mode (Fixed)** - Detection is locked to an accuracy-focused profile
- **Interactive Setup** - GUI wizard to select camera and configure door detection line
- **Flexible Camera Support** - Works with webcams, IP cameras (RTSP), or local video files
- **Smart Line Selection** - Click to set the door line for accurate entry/exit counting
- **Real-Time Tracking** - Track individual people with unique IDs
- **Live Statistics** - See IN/OUT counts and current tracking status
- **Backend Fallback Safety** - Uses custom YOLOv11 when available, otherwise auto-falls back to Ultralytics YOLO11

## 📋 Requirements

- Python 3.8+
- Webcam or IP Camera (RTSP supported)
- 4GB+ RAM (8GB+ recommended for smooth operation)

## 🛠️ Installation

```bash
# Clone or navigate to project
cd live-people-counter

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If custom backend is unavailable, first run may download Ultralytics YOLO11 weights automatically.

## 🚀 Usage

Simply run:
```bash
python main.py
```

## 🎬 Demo Video

See the recorded output here:

- [output.mp4](output.mp4)

If GitHub does not inline-play in your view, click the link to open/download the video.

### Setup Wizard

1. **Select Camera Source**
   - Choose available webcam (0, 1, 2, etc.)
   - Or enter IP camera RTSP URL
   - Or pick a local video file for playback-based counting

2. **Select Door Line + IN Side**
   - Live camera feed appears
   - Drag to draw where people cross the door
   - Press SPACE to lock the line
   - Click the side that means IN
   - Press SPACE to confirm, ESC to cancel/redraw

3. **Counting Starts**
   - Green line shows where door is
   - Watch real-time IN/OUT counts
   - Press ESC to stop

## 📁 Project Structure

```
.
├── main.py           # Entry point and main loop
├── config.py         # Configuration classes
├── detector.py       # YOLOv11 wrapper (custom backend + fallback)
├── tracker.py        # Person tracking and line crossing
├── setup_flow.py     # Setup wizard flow (prompt + line drawing)
├── output.mp4        # Recorded demo output video
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── LICENSE           # MIT License
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Accurate mode is fixed in code.
# Optional custom YOLOv11 integration:
detection_config.prefer_custom_yolov11 = True
detection_config.custom_yolov11_repo_path = r"F:\path\to\YOLO"
detection_config.custom_yolov11_weights = r"F:\path\to\YOLO\weights\yolov11n.pt"
```

### Custom YOLOv11 Setup (B4rtekk1)

1. Clone [B4rtekk1/YOLO](https://github.com/B4rtekk1/YOLO) locally.
2. Put compatible weights in its weights folder (for example yolov11n.pt).
3. Set repo/weights paths in config.py.
4. Start app. Logs will show whether custom backend is active or fallback is used.

## 🔧 How It Works

### Detection Pipeline

```
Camera Feed
    ↓
[Skip Frames] → Resize (0.5x) → YOLOv11 Detect
    ↓
[Match to Tracks] → Check Line Crossing
    ↓
[Count & Log] → Display Stats
```

### Tracking System

Each detected person gets:
- **Unique ID** - For consistent tracking
- **Position History** - Last detection location
- **Counted Flag** - Prevent double counting
- **Age** - Remove lost tracks after inactivity

### Door Line Crossing

```
        ↑
   OUTSIDE
        |
   ━━━━━━━━━━━━━━  ← DOOR LINE (you click here)
        |
     INSIDE
        ↓
```

When a person crosses:
- **From outside to inside** = IN (counter +1)
- **From inside to outside** = OUT (counter +1)
- Counted only once per crossing

## 🎯 Example Use Cases

### Store Entrance
```
Position camera at doorway
Count customers entering/leaving
Track occupancy levels
```

### Office/Building
```
Monitor who entered which room
Track daily foot traffic
Peak hour analysis
```

### Conference Room
```
Track current occupancy
Ensure capacity limits
Auto-lock when full
```

## 🐛 Troubleshooting

### "Failed to open camera"
- Check camera is connected
- Try different camera index (0, 1, 2...)
- For IP camera: verify RTSP URL is correct

### Laggy/Slow Performance
- Use fallback backend (Ultralytics YOLO11) if custom repo/weights are heavy or missing
- Run on lower camera resolution (for webcam, set camera output to 720p)
- Close other programs

### People Counted Multiple Times
- Adjust door line position (should be in middle of crossing area)
- Increase zone size in tracker parameters
- Lower confidence threshold slightly

### IP Camera Connection Issues
```
Example RTSP URLs:
- Hikvision: rtsp://user:pass@192.168.1.100:554/Streaming/Channels/101
- Dahua: rtsp://user:pass@192.168.1.100:554/stream
- Generic: rtsp://192.168.1.100:554/stream
```

### GPU/CUDA Setup (Faster Processing)
```bash
# Install GPU-accelerated PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 📊 Output

### Console Logs
```
INFO | Person 1 went in at 14:32:45
INFO | Person 2 went in at 14:33:10
INFO | Person 1 went out at 14:35:22
```

### Display Stats
- **FPS** - Current frames per second (target: 20+)
- **People IN** - Total entered
- **People OUT** - Total exited
- **Currently Tracking** - Active person tracks

## 🤝 Contributing

Contributions welcome! To improve:
1. Fork the repo
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m 'Add improvement'`
4. Push: `git push origin feature/improvement`
5. Open Pull Request

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Credits

- **YOLOv11 Custom Implementation** by [B4rtekk1/YOLO](https://github.com/B4rtekk1/YOLO)
- **YOLO11 Fallback Runtime** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning framework

## 👨‍💻 Author

**Subodh Adhikari**

GitHub: [@adhikarisubodh9991](https://github.com/adhikarisubodh9991)

---

<p align="center">
Made for real-time computer vision applications 🎯
</p>
