"""
Local web app for unauthorized person detection.

Run:
    python app.py                       # webcam 0, http://127.0.0.1:5000
    python app.py --camera 1            # another webcam
    python app.py --camera "rtsp://user:pass@192.168.0.10/stream"   # IP camera
    python app.py --host 0.0.0.0        # open from other devices on your WiFi
"""

import argparse
import time
from pathlib import Path

from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, send_from_directory, url_for)

from core import CameraWorker, FaceEngine

p = argparse.ArgumentParser()
p.add_argument("--data", default="data", help="folder with labels.csv, images/, embeddings.pkl")
p.add_argument("--camera", default="0", help="webcam index or RTSP/video URL")
p.add_argument("--threshold", type=float, default=0.60)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=5000)
args = p.parse_args()

DATA_DIR = Path(args.data)
ALERT_DIR = DATA_DIR / "alerts"

engine = FaceEngine(DATA_DIR, threshold=args.threshold)
camera = CameraWorker(engine, int(args.camera) if args.camera.isdigit() else args.camera,
                      alerts_dir=ALERT_DIR)
camera.start()

app = Flask(__name__)
app.secret_key = "change-me"


def list_alerts(session_only):
    files = sorted(ALERT_DIR.glob("unauthorized_*.jpg"), reverse=True)
    if session_only:
        files = [f for f in files if f.stem.replace("unauthorized_", "") >= camera.session_start]
    out = []
    for f in files:
        ts = f.stem.replace("unauthorized_", "")
        nice = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        out.append({"file": f.name, "time": nice})
    return out


# ---------- pages ----------

@app.route("/")
def index():
    return render_template("index.html", threshold=engine.threshold,
                           people=len(engine.db), page="live")


@app.route("/people", methods=["GET", "POST"])
def people():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        files = request.files.getlist("photos")
        uploads = [(f.filename, f.read()) for f in files if f and f.filename]
        if not name or not uploads:
            flash("Give a name and at least one photo.", "error")
        else:
            added, skipped = engine.add_person(name, uploads)
            if added:
                flash(f"Added {added} photo(s) for {name}.", "ok")
            if skipped:
                flash(f"No face found in: {', '.join(skipped)}", "error")
        return redirect(url_for("people"))
    return render_template("people.html", people=engine.people(), page="people")


@app.route("/people/delete", methods=["POST"])
def delete_person():
    name = request.form.get("name", "")
    engine.delete_person(name)
    flash(f"Removed {name}.", "ok")
    return redirect(url_for("people"))


@app.route("/people/rebuild", methods=["POST"])
def rebuild():
    skipped = engine.rebuild()
    flash(f"Rebuilt from labels.csv. {len(engine.db)} people.", "ok")
    if skipped:
        flash(f"{len(skipped)} image(s) had no face and were skipped.", "error")
    return redirect(url_for("people"))


@app.route("/alerts")
def alerts():
    show_all = request.args.get("all") == "1"
    return render_template("alerts.html", alerts=list_alerts(not show_all),
                           show_all=show_all, page="alerts")


# ---------- media and API ----------

@app.route("/data/images/<path:fname>")
def person_image(fname):
    return send_from_directory(DATA_DIR / "images", fname)


@app.route("/data/alerts/<path:fname>")
def alert_image(fname):
    return send_from_directory(ALERT_DIR, fname)


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            jpg = camera.get_jpeg()
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    s = dict(camera.status)
    s["session_alerts"] = len(list_alerts(True))
    s["threshold"] = engine.threshold
    return jsonify(s)


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    if "threshold" in data:
        engine.threshold = max(0.0, min(1.0, float(data["threshold"])))
    if "paused" in data:
        camera.paused = bool(data["paused"])
    return jsonify(ok=True, threshold=engine.threshold, paused=camera.paused)


if __name__ == "__main__":
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
