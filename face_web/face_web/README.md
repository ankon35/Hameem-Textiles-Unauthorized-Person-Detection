# Face Guard: local web app

Live unauthorized person detection in your browser, running on your own computer.

* **Live**: camera feed with green (authorized) and red (UNAUTHORIZED) boxes, a flashing alert banner, and a threshold slider
* **People**: add a person by name with photos, remove people, rebuild from `labels.csv`
* **Alerts**: snapshots from this session, or all of them

## 1. Install (once)

Use Python 3.10 to 3.12.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

pip install torch torchvision    # CPU build. For an NVIDIA GPU, get the command from pytorch.org
pip install -r requirements.txt
pip install facenet-pytorch --no-deps
```

The first run downloads the FaceNet weights (about 107 MB), once.

## 2. Bring your Colab data (optional)

Download your Drive folder `Face Detection` and copy its content into `data/`:

```
face_web/
  app.py
  core.py
  data/
    labels.csv
    images/
    embeddings.pkl     <- works as is, made by the Colab notebook
    alerts/            <- created automatically
```

Or skip this and add people from the **People** page.

## 3. Run

```bash
python app.py
```

Open http://127.0.0.1:5000

| Option | Example | Meaning |
|---|---|---|
| `--camera` | `--camera 1` | another webcam |
| | `--camera "rtsp://user:pass@192.168.0.10:554/stream"` | IP / CCTV camera |
| `--threshold` | `--threshold 0.65` | starting threshold (also adjustable on the page) |
| `--host` | `--host 0.0.0.0` | open the page from your phone on the same WiFi: `http://<your PC IP>:5000` |
| `--data` | `--data D:/face_data` | data folder somewhere else |

## Notes

* Only one program can use the webcam at a time. Close Zoom, Meet, or the Camera app first.
* With only one registered person, the threshold cannot be estimated from your data. Register 2 or more people with 5 or more photos each, then tune the slider until known people stay green.
* This is for local use. Before putting it on the internet, add a login, since the page shows the camera and the face photos.
* A printed photo or a phone screen of an authorized face will pass. Add a liveness check for real security.
