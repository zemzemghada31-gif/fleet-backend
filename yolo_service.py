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
CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))
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

try:
    _langs = pytesseract.get_languages(config="")
    print(f"[YOLO] Langues Tesseract disponibles: {_langs}")
    if "ara" not in _langs:
        print("[YOLO] ⚠️  Arabe non disponible. Installe le pack 'ara' avec:")
        print("[YOLO]    sudo apt install tesseract-ocr-ara   (Linux)")
        print("[YOLO]    ou télécharge depuis: https://github.com/tesseract-ocr/tessdata_best")
except Exception as e:
    print(f"[YOLO] Impossible de lister les langues Tesseract: {e}")
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
            backends = [cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                self.cap = cv2.VideoCapture(camera_id, backend)
                if self.cap and self.cap.isOpened():
                    break
            if not self.cap or not self.cap.isOpened():
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
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.cap = None
        self.camera_active = False

    # ── Inference ──────────────────────────────────────────────────────────

    def detect_plate(self) -> tuple[Optional[str], Optional[bytes]]:
        frame = self._grab_frame()
        if frame is None:
            return None, None

        return self._detect_plate_pipeline(frame)

    def detect_plate_from_frame(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        return self._detect_plate_pipeline(frame)

    def _detect_plate_pipeline(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        h, w = frame.shape[:2]
        print(f"[PIPELINE] Image {w}x{h}")

        # Étape 1 : Détection CV directe de la plaque (contours, couleur blanche)
        plate_text, plate_image = self._cv_detect(frame)
        if plate_text:
            print(f"[PIPELINE] OK CV detect: [{plate_text}]")
            return plate_text, plate_image
        print("[PIPELINE] CV detect: rien")

        # Étape 2 : Détection YOLO de la voiture → puis plaque dans le crop
        if self.load_model():
            plate_text, plate_image = self._yolo_detect(frame)
            if plate_text:
                print(f"[PIPELINE] OK YOLO: [{plate_text}]")
                return plate_text, plate_image
            print("[PIPELINE] YOLO: rien")
        else:
            print("[PIPELINE] Modele YOLO non charge")

        # Étape 3 : OCR sur l'image entière en dernier recours
        plate_text = self._ocr_read(frame)
        if plate_text:
            print(f"[PIPELINE] OK OCR full: [{plate_text}]")
            _, jpeg = cv2.imencode(".jpg", frame)
            return plate_text, jpeg.tobytes()
        print("[PIPELINE] OCR full: rien → echec")

        return None, None

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
            print("[YOLO] Aucune voiture detectee (best_box=None)")
            return None, None

        x1, y1, x2, y2 = best_box
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None, None

        print(f"[YOLO] Voiture detectee: box=({x1},{y1},{x2},{y2}) crop={cropped.shape} conf={best_conf:.2f}")

        # Étape 1 : chercher la plaque dans le crop de la voiture (détection CV)
        p_text, p_img = self._detect_canny_plate(cropped)
        if not p_text:
            p_text, p_img = self._detect_gradient_plate(cropped)

        if p_text:
            return p_text, p_img

        # Fallback : OCR sur toute la voiture
        _, jpeg = cv2.imencode(".jpg", cropped)
        plate_image = jpeg.tobytes()
        plate_text = self._ocr_read(cropped)
        if plate_text:
            print(f"[YOLO] OCR full voiture: [{plate_text}]")

        return plate_text, plate_image

    def _cv_detect(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        h, w = frame.shape[:2]

        result = self._detect_canny_plate(frame)
        if result[0]:
            return result

        result = self._detect_gradient_plate(frame)
        if result[0]:
            return result

        result = self._detect_white_plate(frame)
        if result[0]:
            return result

        print(f"[YOLO] Aucune plaque detectee dans l'image {w}x{h}")
        return None, None

    def _detect_canny_plate(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        h_im, w_im = frame.shape[:2]
        max_plate_area = h_im * w_im * 0.35

        for low, high in [(30, 200), (50, 150), (20, 100)]:
            bfilter = cv2.bilateralFilter(gray, 9, 17, 17)
            edged = cv2.Canny(bfilter, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edged = cv2.dilate(edged, kernel, iterations=1)

            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300 or area > max_plate_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect = bw / max(1, bh)
                if aspect < 1.2 or aspect > 8.0 or bw < 40 or bh < 12:
                    continue
                candidates.append((area, x, y, bw, bh))

            candidates.sort(key=lambda c: c[0], reverse=True)

            for area, x, y, bw, bh in candidates:
                margin_x = int(bw * 0.15)
                margin_y = int(bh * 0.15)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(frame.shape[1], x + bw + margin_x)
                y2 = min(frame.shape[0], y + bh + margin_y)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                plate_text = self._ocr_read(cropped)
                if plate_text:
                    _, jpeg = cv2.imencode(".jpg", cropped)
                    return plate_text, jpeg.tobytes()

        return None, None

    def _detect_gradient_plate(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        mag = np.uint8(np.clip(mag, 0, 255))

        _, thresh = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        h_im, w_im = frame.shape[:2]
        max_plate_area = h_im * w_im * 0.35

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > max_plate_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1, bh)
            if aspect < 1.5 or aspect > 6.0 or bw < 50 or bh < 15:
                continue
            candidates.append((area, x, y, bw, bh))

        candidates.sort(key=lambda c: c[0], reverse=True)

        for area, x, y, bw, bh in candidates:
            x1 = max(0, x - 5)
            y1 = max(0, y - 5)
            x2 = min(frame.shape[1], x + bw + 5)
            y2 = min(frame.shape[0], y + bh + 5)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            plate_text = self._ocr_read(cropped)
            if plate_text:
                _, jpeg = cv2.imencode(".jpg", cropped)
                return plate_text, jpeg.tobytes()

        return None, None

    def _detect_white_plate(self, frame: np.ndarray) -> tuple[Optional[str], Optional[bytes]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 168])
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_area = 0
        best_box = None
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect = bw / max(1, bh)
            if area < 800 or aspect < 1.5 or aspect > 7.0 or bw < 60 or bh < 20:
                continue
            if area > best_area:
                best_area = area
                best_box = (x, y, bw, bh)

        if best_box:
            x, y, bw, bh = best_box
            margin = 10
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + bw + margin)
            y2 = min(frame.shape[0], y + bh + margin)
            cropped = frame[y1:y2, x1:x2]
            plate_text = self._ocr_read(cropped)
            if plate_text:
                _, jpeg = cv2.imencode(".jpg", cropped)
                return plate_text, jpeg.tobytes()

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

    def _preprocess_plate(self, gray: np.ndarray, fx: int = 3) -> list:
        h, w = gray.shape
        # Redimensionner fortement pour Tesseract
        big = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)

        # Débruitage léger (conserve les bords)
        denoised = cv2.fastNlMeansDenoising(big, h=10)

        # Sharpen (renforce les caractères)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        variants = [
            ("raw", big),
            ("sharp", sharpened),
            ("clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(big)),
        ]

        # Seuillage uniquement si le contraste est bon
        mean_val = np.mean(big)
        if mean_val > 40 and mean_val < 210:
            _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(("otsu", otsu))
            _, otsu_inv = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants.append(("otsu_inv", otsu_inv))

        return variants

    def _tesseract_ocr(self, cropped: np.ndarray) -> Optional[str]:
        try:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            variants = self._preprocess_plate(gray)

            # Configs organisées : PSM 3 (auto) en premier, puis spécifiques
            configs_latin = [
                "--psm 3 -l fra+eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                "--psm 7 -l fra+eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                "--psm 8 -l fra+eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                "--psm 13 -l fra+eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
                "--psm 6 -l fra+eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            ]
            configs_arabic = [
                "--psm 3 -l ara --oem 3",
                "--psm 7 -l ara --oem 3",
                "--psm 8 -l ara --oem 3",
                "--psm 13 -l ara --oem 3",
            ]

            for name, variant in variants:
                for config in configs_latin:
                    text = pytesseract.image_to_string(variant, config=config)
                    cleaned = self._clean_plate(text)
                    if cleaned:
                        print(f"[OCR] OK latin/{name}: [{cleaned}]")
                        return cleaned
                for config in configs_arabic:
                    text = pytesseract.image_to_string(variant, config=config)
                    cleaned = self._clean_plate(text)
                    if cleaned:
                        print(f"[OCR] OK arabe/{name}: [{cleaned}]")
                        return cleaned

            return None
        except Exception as e:
            print(f"[YOLO][OCR] Erreur Tesseract: {e}")
            return None

    def _contour_ocr(self, cropped: np.ndarray) -> Optional[str]:
        try:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10,
            )

            configs = [
                "--psm 3 -l fra+eng --oem 3",
                "--psm 7 -l fra+eng --oem 3",
                "--psm 8 -l fra+eng --oem 3",
                "--psm 13 -l fra+eng --oem 3",
                "--psm 7 -l ara --oem 3",
                "--psm 8 -l ara --oem 3",
                "--psm 13 -l ara --oem 3",
            ]
            for config in configs:
                text = pytesseract.image_to_string(thresh, config=config)
                cleaned = self._clean_plate(text)
                if cleaned:
                    return cleaned

            return None
        except Exception:
            return None

    @staticmethod
    def _clean_plate(text: str) -> Optional[str]:
        cleaned = "".join(c for c in text.strip() if 'A' <= c.upper() <= 'Z' or '0' <= c <= '9').strip().upper()
        if len(cleaned) < 5 or len(cleaned) > 12:
            return None
        letters = sum('A' <= c <= 'Z' for c in cleaned)
        digits = sum('0' <= c <= '9' for c in cleaned)
        if letters < 2 or digits < 1:
            return None
        return cleaned

    # ── High-level scan ────────────────────────────────────────────────────

    def scan_once(self) -> dict:
        if not self.camera_active:
            ok = self.start_camera()
            if not ok:
                return {"plate": None, "image_b64": None,
                        "error": "Caméra non disponible"}
            # Laisser la caméra s'ajuster (exposition, focus, balance blancs)
            for _ in range(10):
                self._grab_frame()
                time.sleep(0.3)

        for attempt in range(15):
            plate, img_bytes = self.detect_plate()
            if plate:
                image_b64 = base64.b64encode(img_bytes).decode() if img_bytes else None
                self._last_detected_plate = plate
                self._last_detected_image = img_bytes
                self._last_detection_time = time.time()
                return {"plate": plate, "image_b64": image_b64, "error": None}
            time.sleep(0.3)

        return {"plate": None, "image_b64": None, "error": "Aucune plaque détectée après plusieurs essais"}

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
