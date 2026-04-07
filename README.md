# Live People Counter

This project counts people crossing a door line in real time using YOLO-based detection and a simple tracking pipeline.

It includes a setup flow so you can choose a camera source, draw a custom line, and select which side should count as IN.

## Main Features

- Accurate-only detection profile (stable defaults)
- Setup wizard for camera selection and line drawing
- Works with webcam, RTSP/IP camera, or local video file
- Real-time ID tracking and IN/OUT counting
- Optional custom YOLOv11 backend with fallback to Ultralytics

## Requirements

- Python 3.8+
- Webcam or IP camera (RTSP)
- 4 GB RAM minimum (8 GB recommended)

## Installation

```bash
cd live-people-counting

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

If custom YOLOv11 is not available, the app falls back to Ultralytics YOLO11.

## Run

```bash
python main.py
```

## Setup Flow

1. Select input source:
- Webcam index
- RTSP URL
- Local video file

2. Draw counting line:
- Drag to draw the door line
- Press SPACE to lock line
- Click the side that means IN
- Press SPACE to confirm

3. Start counting:
- IN/OUT counters update live
- Press ESC to exit

## Configuration

Most tuning values are in `config.py`.

Example custom backend config:

```python
detection_config.prefer_custom_yolov11 = True
detection_config.custom_yolov11_repo_path = r"F:\path\to\YOLO"
detection_config.custom_yolov11_weights = r"F:\path\to\YOLO\weights\yolov11n.pt"
```

## Build (Windows)

```bash
python -m PyInstaller live-people-counting.spec
```

Latest builds are published in the GitHub Releases tab.

## Troubleshooting

### Camera does not open

- Check camera connection
- Try a different webcam index (0, 1, 2)
- Verify RTSP URL format and credentials

### Slow performance

- Use smaller input resolution (especially for webcam)
- Close heavy background apps
- Use CUDA-enabled PyTorch if you have an NVIDIA GPU

### Counting is inaccurate

- Redraw the line in a cleaner crossing area
- Make sure the IN side is selected correctly
- Avoid extreme camera angles if possible

## Project Files

```
main.py           # app entry point + runtime loop
config.py         # camera and detection configuration
detector.py       # detector wrapper (custom/fallback backend)
tracker.py        # person tracking and crossing logic
setup_flow.py     # setup wizard for source + line
live-people-counting.spec  # PyInstaller build spec
```

## License

MIT. See `LICENSE`.

## Credits

- Custom YOLOv11 option: [B4rtekk1/YOLO](https://github.com/B4rtekk1/YOLO)
- Fallback runtime: [Ultralytics](https://github.com/ultralytics/ultralytics)
