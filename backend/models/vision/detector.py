"""
YOLOv8 Currency Note Detector + Region Cropper.

Detects currency notes in images and crops regions of interest
(security thread, micro-lettering, serial number, watermark, etc.)
for downstream forgery classification.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import config

logger = logging.getLogger(__name__)


CURRENCY_REGIONS = {
    "security_thread": {"x1": 0.35, "y1": 0.05, "x2": 0.40, "y2": 0.95},
    "micro_lettering": {"x1": 0.15, "y1": 0.30, "x2": 0.85, "y2": 0.45},
    "serial_number": {"x1": 0.05, "y1": 0.05, "x2": 0.40, "y2": 0.20},
    "watermark": {"x1": 0.60, "y1": 0.10, "x2": 0.95, "y2": 0.70},
    "latent_image": {"x1": 0.05, "y1": 0.55, "x2": 0.20, "y2": 0.80},
    "color_shifting_ink": {"x1": 0.75, "y1": 0.70, "x2": 0.95, "y2": 0.95},
    "portrait": {"x1": 0.45, "y1": 0.10, "x2": 0.80, "y2": 0.85},
    "denomination": {"x1": 0.05, "y1": 0.70, "x2": 0.25, "y2": 0.95},
}


class CurrencyDetector:
    """
    YOLOv8-based currency note detector with region-of-interest extraction.
    """

    def __init__(self):
        self._model = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load YOLOv8 model."""
        if self._initialized:
            return

        logger.info(" Initializing YOLOv8 currency detector...")
        try:
            from ultralytics import YOLO

            self._model = YOLO("yolov8n.pt")
            self._initialized = True
            logger.info(" YOLOv8n loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {e}")
            raise

    def detect_currency(self, image: np.ndarray) -> dict:
        """Detect currency note in the image and extract regions."""
        if not self._initialized:
            raise RuntimeError("Detector not initialized")

        result = {
            "detected": False,
            "confidence": 0.0,
            "note_bbox": None,
            "note_crop": None,
            "regions": {},
            "note_dimensions": None,
        }

        note_crop = None
        best_confidence = 0.0
        detections = self._model(
            image,
            conf=config.vision_agent.yolo_confidence,
            iou=config.vision_agent.yolo_iou,
            verbose=False,
        )

        if detections and len(detections[0].boxes) > 0:
            best_area = 0
            for box in detections[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                conf = float(box.conf[0])
                if area > best_area:
                    best_area = area
                    best_confidence = conf
                    result["note_bbox"] = [int(x1), int(y1), int(x2), int(y2)]
                    note_crop = image[int(y1) : int(y2), int(x1) : int(x2)]

        if note_crop is None:
            note_crop, bbox = self._detect_by_edges(image)
            if note_crop is not None:
                result["note_bbox"] = bbox
                best_confidence = 0.7

        if note_crop is None:
            note_crop = image.copy()
            best_confidence = 0.3
            result["note_bbox"] = [0, 0, image.shape[1], image.shape[0]]

        result["detected"] = True
        result["confidence"] = round(best_confidence, 4)
        result["note_crop"] = note_crop
        result["note_dimensions"] = {
            "width": note_crop.shape[1],
            "height": note_crop.shape[0],
        }
        result["regions"] = self._extract_regions(note_crop)
        return result

    def _detect_by_edges(
        self, image: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[list]]:
        """Use edge detection + contour finding to locate the note."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        image_area = image.shape[0] * image.shape[1]
        if area < image_area * 0.1:
            return None, None

        x, y, w, h = cv2.boundingRect(largest)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            crop = self._four_point_transform(image, approx.reshape(4, 2))
        else:
            crop = image[y : y + h, x : x + w]

        return crop, [x, y, x + w, y + h]

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform using 4 corner points."""
        rect = self._order_points(pts.astype(np.float32))
        tl, tr, br, bl = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        point_sum = pts.sum(axis=1)
        rect[0] = pts[np.argmin(point_sum)]
        rect[2] = pts[np.argmax(point_sum)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _extract_regions(self, note_crop: np.ndarray) -> dict[str, np.ndarray]:
        """Extract regions of interest from the cropped note image."""
        h, w = note_crop.shape[:2]
        regions = {}

        for region_name, coords in CURRENCY_REGIONS.items():
            x1 = int(coords["x1"] * w)
            y1 = int(coords["y1"] * h)
            x2 = int(coords["x2"] * w)
            y2 = int(coords["y2"] * h)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            region = note_crop[y1:y2, x1:x2]
            if region.size > 0:
                regions[region_name] = region

        return regions

    def get_stats(self) -> dict:
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "model": "YOLOv8n (ultralytics)",
            "regions": list(CURRENCY_REGIONS.keys()),
        }


_detector: Optional[CurrencyDetector] = None


def get_currency_detector() -> CurrencyDetector:
    global _detector
    if _detector is None:
        _detector = CurrencyDetector()
    return _detector
