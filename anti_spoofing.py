import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO

from config import LIVENESS_THRESHOLD, BASE_DIR, get_yolo_device, get_torch_device

log = logging.getLogger(__name__)

class AntiSpoofing:
    """
    A robust anti-spoofing mechanism that uses YOLOv8 to detect if a face 
    is located inside a mobile phone, laptop, or TV/Monitor screen.
    This directly aligns with the requirement to only flag spoofs if shown on a device.
    """
    def __init__(self):
        # Load YOLOv8 nano (very fast, already used in office_tracker)
        # COCO Classes: 62 = tv, 63 = laptop, 67 = cell phone
        self.spoof_classes = [62, 63, 67]
        self._yolo_device = get_yolo_device()
        try:
            self.yolo = YOLO("yolov8n.pt")
            self.last_frame_id = None
            self.last_results = None
            log.info(
                "[SPOOF] YOLOv8 Anti-spoofing model loaded on %s.",
                get_torch_device(),
            )
        except Exception as e:
            log.error(f"[SPOOF] Failed to load YOLO: {e}")
            self.yolo = None

    def available(self):
        return self.yolo is not None

    def check_liveness(self, frame, bbox):
        """
        Check if the face is real by ensuring it is not displayed inside a phone or laptop.
        Returns: (is_real: bool, liveness_score: float)
        """
        if self.yolo is None:
            return True, 1.0
            
        fx1, fy1, fx2, fy2 = bbox
        face_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
        
        if face_area <= 0:
            return True, 1.0

        # Cache YOLO results per frame to avoid running it multiple times if there are multiple faces
        frame_id = id(frame)
        if self.last_frame_id != frame_id:
            self.last_results = self.yolo(
                frame, imgsz=320, verbose=False, device=self._yolo_device,
            )
            self.last_frame_id = frame_id
            
        results = self.last_results
        
        is_real = True
        score = 1.0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.spoof_classes:
                    # Found a phone, laptop, or TV
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate how much of the face is inside the screen
                    ix1 = max(fx1, bx1)
                    iy1 = max(fy1, by1)
                    ix2 = min(fx2, bx2)
                    iy2 = min(fy2, by2)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        intersection = (ix2 - ix1) * (iy2 - iy1)
                        overlap_ratio = intersection / face_area
                        
                        # If more than 30% of the face bounding box overlaps with a screen bounding box,
                        # it's highly likely they are holding a phone/laptop up to the camera!
                        if overlap_ratio > 0.3:
                            is_real = False
                            score = 0.01
                            log.warning(f"[SPOOF] Face is inside object class {cls_id} (Overlap: {overlap_ratio:.2f})")
                            break
            if not is_real:
                break
                
        return is_real, score
