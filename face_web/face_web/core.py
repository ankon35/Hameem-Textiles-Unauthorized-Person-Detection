"""
Face engine: detection (MTCNN), embeddings (FaceNet), the people database,
and the camera worker that runs recognition in the background.
"""

import csv
import io
import pickle
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def l2n(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_") or "person"


class FaceEngine:
    """Holds the models and the authorized people database."""

    def __init__(self, data_dir="data", threshold=0.60, min_prob=0.90):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.csv_path = self.data_dir / "labels.csv"
        self.db_path = self.data_dir / "embeddings.pkl"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = threshold
        self.min_prob = min_prob
        self.lock = threading.Lock()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[engine] device: {self.device}, loading models...")
        self.mtcnn_one = MTCNN(image_size=160, margin=20, select_largest=True, device=self.device)
        self.mtcnn_all = MTCNN(image_size=160, margin=20, keep_all=True, device=self.device)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self.db = {}
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self.db = pickle.load(f)
            print(f"[engine] loaded {len(self.db)} people from {self.db_path}")
        elif self.csv_path.exists():
            self.rebuild()
        self._refresh_flat()

    # ---------- database ----------

    def _refresh_flat(self):
        names = [n for n, e in self.db.items() for _ in range(len(e))]
        self.db_names = np.array(names)
        self.db_embs = np.vstack(list(self.db.values())) if self.db else np.zeros((0, 512))

    def _save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)
        self._refresh_flat()

    def _read_csv(self):
        rows = []
        if not self.csv_path.exists():
            return rows
        with open(self.csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = (row.get("image_path") or "").strip()
                name = (row.get("name") or "").strip()
                if path and name:
                    rows.append((path, name))
        return rows

    def _write_csv(self, rows):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "name"])
            w.writerows(rows)

    def embed_one(self, pil_img):
        face = self.mtcnn_one(pil_img)
        if face is None:
            return None
        with torch.no_grad():
            emb = self.facenet(face.unsqueeze(0).to(self.device)).cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

    def rebuild(self):
        """Rebuild embeddings from labels.csv and the images folder."""
        db = {}
        skipped = []
        for path, name in self._read_csv():
            p = Path(path)
            if not p.is_absolute():
                p = self.data_dir / p
            try:
                emb = self.embed_one(Image.open(p).convert("RGB"))
            except Exception:
                emb = None
            if emb is None:
                skipped.append(str(p))
                continue
            db.setdefault(name, []).append(emb)
        with self.lock:
            self.db = {n: np.stack(e) for n, e in db.items()}
            self._save()
        print(f"[engine] rebuilt: {len(self.db)} people, {len(skipped)} image(s) skipped")
        return skipped

    def add_person(self, name, uploads):
        """uploads: list of (filename, bytes). Returns (added, skipped_filenames)."""
        name = name.strip()
        rows = self._read_csv()
        new_embs, added, skipped = [], 0, []
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for i, (fname, data) in enumerate(uploads):
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                skipped.append(fname)
                continue
            emb = self.embed_one(img)
            if emb is None:
                skipped.append(fname)
                continue
            out_name = f"{safe_name(name)}_{stamp}_{i:02d}.jpg"
            img.save(self.images_dir / out_name, quality=92)
            rows.append((f"images/{out_name}", name))
            new_embs.append(emb)
            added += 1
        if added:
            self._write_csv(rows)
            with self.lock:
                old = self.db.get(name)
                stack = np.stack(new_embs)
                self.db[name] = stack if old is None else np.vstack([old, stack])
                self._save()
        return added, skipped

    def delete_person(self, name):
        rows = self._read_csv()
        keep = []
        for path, n in rows:
            if n == name:
                p = self.data_dir / path
                if p.exists() and p.parent == self.images_dir:
                    p.unlink()
            else:
                keep.append((path, n))
        self._write_csv(keep)
        with self.lock:
            self.db.pop(name, None)
            self._save()

    def people(self):
        counts = {}
        for path, name in self._read_csv():
            counts.setdefault(name, {"images": 0, "faces": 0, "thumb": None})
            counts[name]["images"] += 1
            if counts[name]["thumb"] is None:
                counts[name]["thumb"] = path
        for name, e in self.db.items():
            counts.setdefault(name, {"images": 0, "faces": 0, "thumb": None})
            counts[name]["faces"] = len(e)
        return dict(sorted(counts.items()))

    # ---------- recognition ----------

    def recognize(self, rgb):
        """rgb: HxWx3 RGB numpy array. Returns list of (box, name or None, score)."""
        img = Image.fromarray(rgb)
        boxes, probs = self.mtcnn_all.detect(img)
        if boxes is None:
            return []
        keep = [i for i, p in enumerate(probs) if p is not None and p >= self.min_prob]
        if not keep:
            return []
        boxes = boxes[keep]
        faces = self.mtcnn_all.extract(img, boxes, save_path=None)
        if faces.dim() == 3:
            faces = faces.unsqueeze(0)
        with torch.no_grad():
            embs = l2n(self.facenet(faces.to(self.device)).cpu().numpy())
        with self.lock:
            names, db_embs, thr = self.db_names, self.db_embs, self.threshold
        out = []
        if len(db_embs) == 0:
            return [(b, None, 0.0) for b in boxes]
        sims = embs @ db_embs.T
        for box, row in zip(boxes, sims):
            j = int(row.argmax())
            s = float(row[j])
            out.append((box, str(names[j]) if s >= thr else None, s))
        return out


def draw(bgr, results):
    for box, name, s in results:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 200, 0) if name else (0, 0, 255)
        label = f"{name} {s:.2f}" if name else f"UNAUTHORIZED {s:.2f}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return bgr


class CameraWorker(threading.Thread):
    """Reads the camera, runs recognition, keeps the latest annotated JPEG, saves alerts."""

    def __init__(self, engine, source=0, alerts_dir="data/alerts",
                 confirm_frames=5, cooldown=10.0, every=2):
        super().__init__(daemon=True)
        self.engine = engine
        self.source = source
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_frames = confirm_frames
        self.cooldown = cooldown
        self.every = every

        self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.latest_jpeg = None
        self.status = {"state": "starting", "faces": [], "alert": False, "fps": 0.0}
        self.running = True
        self.paused = False
        self.frame_lock = threading.Lock()

    def save_alert(self, bgr, score):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.alerts_dir / f"unauthorized_{ts}.jpg"
        cv2.imwrite(str(path), bgr)
        log = self.alerts_dir / "log.csv"
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "image", "best_similarity"])
            w.writerow([ts, path.name, f"{score:.3f}"])
        print(f"[ALERT] {ts} -> {path}")

    def _placeholder(self, text):
        img = np.full((480, 640, 3), 30, np.uint8)
        cv2.putText(img, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status["state"] = f"cannot open camera {self.source}"
            self.latest_jpeg = self._placeholder(f"Cannot open camera {self.source}")
            print(f"[camera] cannot open {self.source}")
            return

        results, streak, last_alert, i = [], 0, 0.0, 0
        t0 = time.time()
        while self.running:
            if self.paused:
                self.latest_jpeg = self._placeholder("Paused")
                self.status.update(state="paused", faces=[], alert=False)
                time.sleep(0.2)
                continue

            ok, frame = cap.read()
            if not ok:
                self.status["state"] = "camera frame not received"
                time.sleep(0.5)
                continue

            if i % self.every == 0:
                results = self.engine.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                unknown = [s for _, n, s in results if n is None]
                streak = streak + 1 if unknown else 0
                if streak >= self.confirm_frames and time.time() - last_alert > self.cooldown:
                    self.save_alert(draw(frame.copy(), results), min(unknown))
                    last_alert = time.time()

            annotated = draw(frame, results)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()

            i += 1
            if i % 15 == 0:
                self.status["fps"] = round(15 / (time.time() - t0), 1)
                t0 = time.time()
            self.status.update(
                state="running",
                alert=streak >= self.confirm_frames,
                faces=[{"name": n, "score": round(s, 2)} for _, n, s in results],
            )

        cap.release()

    def get_jpeg(self):
        with self.frame_lock:
            return self.latest_jpeg
