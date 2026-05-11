import asyncio
import base64
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import onnxruntime as ort


# ─── Configuration ───────────────────────────────────────────────────────────
CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE", "0.4"))
COOLDOWN_SECONDS = float(os.environ.get("YOLO_COOLDOWN", "5.0"))
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8n.onnx"

# Tesseract OCR
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
os.environ["TESSERACT_CMD"] = TESSERACT_CMD
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# ──────────────────────────────────────────────────────────────────────────────


class YoloService:
    _instance: Optional["YoloService"] = None

    @classmethod
    def get_instance(cls) -> "YoloService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.session: Optional[ort.InferenceSession] = None
        self.camera_active = False
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_detection_time = 0.0
        self._last_detected_plate: Optional[str] = None
        self._last_detected_image: Optional[bytes] = None
        self._current_frame: Optional[bytes] = None
        self._on_detection_callback: Optional[Callable] = None

    # ── Model management ──────────────────────────────────────────────────

    def is_model_available(self) -> bool:
        return MODEL_PATH.exists()

    def ensure_model(self) -> bool:
        if self.is_model_available():
            return True
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urls = [
            "https://huggingface.co/Xuban/yolo_weights_database/resolve/main/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
            "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx",
        ]
        for url in urls:
            try:
                import urllib.request
                print(f"[YOLO] Téléchargement du modèle: {url}")
                urllib.request.urlretrieve(url, str(MODEL_PATH))
                print(f"[YOLO] Modèle téléchargé: {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"[YOLO] Échec: {e}")
        return False

    def load_model(self) -> bool:
        if self.session is not None:
            return True
        if not self.is_model_available():
            ok = self.ensure_model()
            if not ok:
                return False
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 2
            self.session = ort.InferenceSession(
                str(MODEL_PATH), so,
                providers=["CPUExecutionProvider"],
            )
            return True
        except Exception as e:
            print(f"[YOLO] Erreur chargement modèle: {e}")
            return False

    # ── Camera management ──────────────────────────────────────────────────

    def start_camera(self, camera_id: int = CAMERA_ID) -> bool:
        if self.camera_active:
            return True
        try:
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                self.cap = None
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_active = True
            return True
        except Exception:
            self.camera_active = False
            return False

    def stop_camera(self):
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.camera_active = False

    # ── Inference ──────────────────────────────────────────────────────────

    def detect_plate(self) -> tuple[Optional[str], Optional[bytes]]:
        frame = self._grab_frame()
        if frame is None:
            return None, None

        plate_text = None
        plate_image = None

        if self.load_model():
            plate_text, plate_image = self._yolo_detect(frame)

        if plate_text is None:
            plate_text, plate_image = self._cv_detect(frame)

        return plate_text, plate_image

    def detect_plate_from_frame(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        plate_text = None
        plate_image = None

        if self.load_model():
            plate_text, plate_image = self._yolo_detect(frame)

        if plate_text is None:
            plate_text, plate_image = self._cv_detect(frame)

        return plate_text, plate_image

    def _yolo_detect(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        if self.session is None:
            return None, None

        h, w = frame.shape[:2]
        input_shape = (640, 640)
        resized = cv2.resize(frame, input_shape)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        try:
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        except Exception:
            return None, None

        detections = outputs[0][0]
        best_box = None
        best_conf = 0.0

        for det in detections:
            scores = det[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            if cls_id not in (0, 1, 2, 3, 5, 7):
                continue

            xc, yc, bw, bh = det[:4]
            x1 = int((xc - bw / 2) * w / 640)
            y1 = int((yc - bh / 2) * h / 640)
            x2 = int((xc + bw / 2) * w / 640)
            y2 = int((yc + bh / 2) * h / 640)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            aspect = (x2 - x1) / max(1, (y2 - y1))

            if area < 500 or aspect < 1.5 or aspect > 6.0:
                continue

            if conf > best_conf:
                best_conf = conf
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            return None, None

        x1, y1, x2, y2 = best_box
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None, None

        _, jpeg = cv2.imencode(".jpg", cropped)
        plate_image = jpeg.tobytes()
        plate_text = self._ocr_read(cropped)

        return plate_text, plate_image

    def _cv_detect(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, bw, bh = cv2.boundingRect(approx)
            aspect = bw / max(1, bh)
            if aspect < 1.5 or aspect > 6.0 or bw < 50 or bh < 15:
                continue

            cropped = frame[y:y + bh, x:x + bw]
            if cropped.size == 0:
                continue

            _, jpeg = cv2.imencode(".jpg", cropped)
            plate_image = jpeg.tobytes()
            plate_text = self._ocr_read(cropped)

            return plate_text, plate_image

        return None, None

    def _grab_frame(self) -> Optional[np.ndarray]:
        if self.cap and self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                return frame
        test_img = os.environ.get("YOLO_TEST_IMAGE", "")
        if test_img and os.path.exists(test_img):
            return cv2.imread(test_img)
        return None

    # ── OCR ────────────────────────────────────────────────────────────────

    def _ocr_read(self, cropped: np.ndarray) -> Optional[str]:
        text = self._tesseract_ocr(cropped)
        if text:
            return text
        text = self._contour_ocr(cropped)
        return text

    def _tesseract_ocr(self, cropped: np.ndarray) -> Optional[str]:
        try:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text = pytesseract.image_to_string(thresh, config=config)
            cleaned = self._clean_plate(text)
            if cleaned:
                return cleaned
            config2 = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text2 = pytesseract.image_to_string(thresh, config=config2)
            return self._clean_plate(text2)
        except Exception:
            return None

    def _contour_ocr(self, cropped: np.ndarray) -> Optional[str]:
        try:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if 20 < cv2.contourArea(c) < 5000]

            bounds = [cv2.boundingRect(c) for c in contours]
            bounds.sort(key=lambda b: b[0])

            chars = []
            for x, y, w, h in bounds:
                aspect = h / max(1, w)
                if 1.2 < aspect < 6.0 and h > 20:
                    char_roi = thresh[y:y + h, x:x + w]
                    ratio = h / 50.0
                    resized = cv2.resize(char_roi, (int(25 * ratio), 50))
                    chars.append(resized)

            if len(chars) >= 3:
                return "?PLAQUE?"
            return None
        except Exception:
            return None

    @staticmethod
    def _clean_plate(text: str) -> Optional[str]:
        cleaned = "".join(c for c in text.strip() if c.isalnum() or c in "- ").strip()
        cleaned = cleaned.upper()
        if "-" not in cleaned and len(cleaned) >= 6:
            cleaned = cleaned[:3] + "-" + cleaned[3:]
        if cleaned and 4 <= len(cleaned) <= 12:
            return cleaned
        return None

    # ── High-level scan ────────────────────────────────────────────────────

    def scan_once(self) -> dict:
        if not self.camera_active:
            ok = self.start_camera()
            if not ok:
                return {"plate": None, "image_b64": None,
                        "error": "Caméra non disponible"}
            time.sleep(0.5)

        plate, img_bytes = self.detect_plate()
        image_b64 = base64.b64encode(img_bytes).decode() if img_bytes else None
        self._last_detected_plate = plate
        self._last_detected_image = img_bytes

        if plate:
            self._last_detection_time = time.time()

        return {"plate": plate, "image_b64": image_b64, "error": None}

    # ── Continuous monitoring ──────────────────────────────────────────────

    def start_monitoring(self, callback: Optional[Callable] = None):
        if self.monitoring:
            return
        self.monitoring = True
        self._on_detection_callback = callback
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    def stop_monitoring(self):
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

    async def _monitor_loop(self):
        if not self.camera_active:
            self.start_camera()
            await asyncio.sleep(0.5)

        while self.monitoring:
            now = time.time()
            if now - self._last_detection_time < COOLDOWN_SECONDS:
                await asyncio.sleep(0.5)
                continue

            plate, img_bytes = self.detect_plate()
            if plate:
                self._last_detected_plate = plate
                self._last_detected_image = img_bytes
                self._last_detection_time = now
                image_b64 = (
                    base64.b64encode(img_bytes).decode() if img_bytes else None
                )
                if self._on_detection_callback:
                    await self._on_detection_callback(plate, image_b64)
            await asyncio.sleep(0.3)

    def get_status(self) -> dict:
        return {
            "camera_active": self.camera_active,
            "camera_id": os.environ.get("CAMERA_ID", "0"),
            "monitoring": self.monitoring,
            "model_loaded": self.session is not None,
            "model_available": self.is_model_available(),
            "last_detected_plate": self._last_detected_plate,
            "last_detection_time": (
                datetime.fromtimestamp(self._last_detection_time).isoformat()
                if self._last_detection_time else None
            ),
        }


yolo_service = YoloService.get_instance()
