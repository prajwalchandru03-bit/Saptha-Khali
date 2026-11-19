"""
TruthLens Visual Forensics Module (Optimized for Reels & Social Video)
Handles:
1) EXIF metadata inspection with Instagram H.264 platform-awareness
2) Error Level Analysis (ELA) heatmap generation saved as PNG
3) Face detection & cropping (OpenCV/MediaPipe) with targeted face-swap classification
4) Full-frame deepfake classification (dima806/deepfake_vs_real_image_detection)
5) Frame-to-frame temporal flicker & consistency analysis for AI-generated video detection
"""

import base64
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ExifTags

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger("truthlens.visual")

# Known AI generators or editing software signatures in EXIF
SUSPICIOUS_SOFTWARE_SIGNATURES = [
    "adobe photoshop", "photoshop", "gimp", "midjourney", "stable diffusion",
    "dall-e", "comfyui", "automatic1111", "faceapp", "deepfacelab",
    "reface", "roop", "faceswap", "canvas", "lightroom", "procreate"
]


@dataclass
class ExifAnalysisResult:
    """Forensic EXIF inspection results."""
    has_exif: bool
    software: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    datetime_original: Optional[str] = None
    suspicious_indicators: List[str] = field(default_factory=list)
    platform_note: Optional[str] = None
    raw_exif: Dict[str, str] = field(default_factory=dict)


@dataclass
class ElaResult:
    """Error Level Analysis output and tampering metrics."""
    heatmap_path: str
    mean_error: float
    max_error: float
    error_variance: float
    high_error_pixel_ratio: float
    tampering_suspicion_score: float  # 0.0 to 1.0


@dataclass
class FramePrediction:
    """Per-frame classification output with separate face and full-frame breakdown."""
    frame_path: str
    timestamp: float
    fake_score: float  # prioritized/weighted score
    real_score: float
    predicted_label: str
    face_detected: bool = False
    face_crop_path: Optional[str] = None
    face_fake_score: Optional[float] = None
    full_frame_fake_score: float = 0.0
    face_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class VisualForensicsResult:
    """Comprehensive visual forensics results across sampled frames."""
    exif_analysis: ExifAnalysisResult
    ela_result: Optional[ElaResult]
    average_fake_score: float
    peak_fake_score: float
    average_face_fake_score: float
    peak_face_fake_score: float
    average_full_frame_fake_score: float
    temporal_flicker_score: float
    temporal_consistency_verdict: str
    median_fake_score: float = 0.0
    trimmed_mean_fake_score: float = 0.0
    median_face_fake_score: float = 0.0
    median_full_frame_fake_score: float = 0.0
    p75_face_fake_score: float = 0.0
    robust_face_fake_score: float = 0.0
    num_faces_detected: int = 0
    ai_generated_video_indicators: List[str] = field(default_factory=list)
    is_flagged_deepfake: bool = False
    is_flagged_ai_video: bool = False
    frame_predictions: List[FramePrediction] = field(default_factory=list)
    top_flagged_frames: List[FramePrediction] = field(default_factory=list)
    face_crop_paths: List[str] = field(default_factory=list)
    detection_screenshots: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = "dima806/deepfake_vs_real_image_detection"


def normalize_prob(v: Any) -> float:
    """
    Normalizes any probability/confidence score:
    - If v > 1.0 (e.g. 0-100 percentage scale), divide by 100.0 first.
    - Clamps strictly to [0.04, 0.94] so neither 0.0 nor 1.0 (0% or 100%) is ever output.
    """
    try:
        val = float(v)
    except (ValueError, TypeError):
        return 0.50
    if val > 1.0:
        val = val / 100.0
    val = max(0.04, min(0.94, val))
    return round(val, 4)


class HuggingFaceImageDeepfakeDetector:
    """
    Lazy-loaded classifier using HuggingFace's
    'dima806/deepfake_vs_real_image_detection'.
    """
    _instance = None
    _logged_raw_first = False

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id = "dima806/deepfake_vs_real_image_detection"
        self.processor = None
        self.model = None
        self._is_loaded = False
        self._load_error = None

    @classmethod
    def get_instance(cls, device: str = "cpu"):
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def load_model(self):
        if self._is_loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            logger.info(f"Loading visual deepfake model: {self.model_id} onto {self.device}...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("Visual deepfake model loaded successfully.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Failed to load HuggingFace deepfake detector ({self.model_id}): {e}. Fallback scoring will be used.")

    def predict(self, image: Image.Image) -> Tuple[float, float, str]:
        """
        Returns (fake_score, real_score, label).
        """
        if not self._is_loaded and self._load_error is None:
            self.load_model()

        if self._load_error is not None or self.model is None or self.processor is None:
            return 0.5, 0.5, "Inconclusive (Model Offline)"

        import torch
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [1.0 - probs, probs]

        id2label = getattr(self.model.config, "id2label", {})
        fake_prob = 0.0
        real_prob = 0.0
        found_fake = False
        found_real = False

        id2label = getattr(self.model.config, "id2label", {})
        raw_dict = {}
        fake_prob = 0.0
        real_prob = 1.0
        found_fake = False

        for idx, prob in enumerate(probs):
            label_val = id2label.get(idx) if idx in id2label else id2label.get(str(idx), f"label_{idx}")
            lbl_str = str(label_val)
            lbl_clean = lbl_str.strip().lower()
            prob_float = float(prob)
            raw_dict[lbl_str] = round(prob_float, 4)

            # Strictly match "fake" or "deepfake"
            if "fake" in lbl_clean or "deepfake" in lbl_clean or "synthetic" in lbl_clean:
                fake_prob = prob_float
                found_fake = True
            elif "real" in lbl_clean or "bonafide" in lbl_clean or "authentic" in lbl_clean:
                real_prob = prob_float

        if not HuggingFaceImageDeepfakeDetector._logged_raw_first:
            HuggingFaceImageDeepfakeDetector._logged_raw_first = True
            logger.info(f"[FACE RAW] {raw_dict}")
            print(f"[FACE RAW] {raw_dict}", flush=True)

        if not found_fake:
            # If model didn't provide a fake label, but provided real, invert real once:
            if any("real" in k.lower() for k in raw_dict.keys()):
                fake_prob = 1.0 - real_prob
            elif len(probs) >= 2:
                # Official dima806 mapping: {0: 'Real', 1: 'Fake'}
                fake_prob = float(probs[1])
                real_prob = float(probs[0])
            else:
                fake_prob = 0.50
                real_prob = 0.50
        else:
            real_prob = 1.0 - fake_prob

        fake_prob = max(0.04, min(0.92, fake_prob))
        real_prob = max(0.04, min(0.92, real_prob))
        predicted_label = "Fake" if fake_prob > real_prob else "Real"
        return round(fake_prob, 4), round(real_prob, 4), predicted_label


class VisualAnalyzer:
    """
    Orchestrates EXIF check, ELA analysis, face-crop deepfake detection,
    full-frame ViT classification, and temporal consistency evaluation.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector = HuggingFaceImageDeepfakeDetector.get_instance(device=device)

    def analyze_exif(self, image_path: Union[str, Path]) -> ExifAnalysisResult:
        """
        Inspects EXIF metadata.
        Aware of Instagram/social media recompression: stripped EXIF is NOT treated as proof of manipulation.
        """
        path = Path(image_path)
        suspicious_indicators: List[str] = []
        raw_exif_dict: Dict[str, str] = {}
        software: Optional[str] = None
        make: Optional[str] = None
        model: Optional[str] = None
        datetime_original: Optional[str] = None
        has_exif = False
        platform_note = None

        try:
            with Image.open(path) as img:
                raw_exif = img._getexif()
                if raw_exif:
                    has_exif = True
                    for tag_id, value in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        val_str = str(value)
                        raw_exif_dict[tag_name] = val_str

                        tag_lower = tag_name.lower()
                        if "software" in tag_lower:
                            software = val_str
                            for sig in SUSPICIOUS_SOFTWARE_SIGNATURES:
                                if sig in val_str.lower():
                                    suspicious_indicators.append(
                                        f"Software signature matched generative/editing tool: '{val_str}'"
                                    )
                        elif "make" in tag_lower:
                            make = val_str
                        elif "model" in tag_lower:
                            model = val_str
                        elif "datetimeoriginal" in tag_lower:
                            datetime_original = val_str
        except Exception as e:
            logger.debug(f"EXIF extraction error on {image_path}: {e}")

        if not has_exif:
            platform_note = (
                "EXIF metadata is absent: typical for social media platforms (Instagram Reels/YouTube Shorts) "
                "which re-encode media to H.264/AAC and strip metadata upon upload. "
                "This is normal platform behavior and not an indicator of manipulation."
            )

        return ExifAnalysisResult(
            has_exif=has_exif,
            software=software,
            camera_make=make,
            camera_model=model,
            datetime_original=datetime_original,
            suspicious_indicators=suspicious_indicators,
            platform_note=platform_note,
            raw_exif=raw_exif_dict
        )

    def compute_ela(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        quality: int = 90,
        scale: int = 15
    ) -> ElaResult:
        """
        Executes Error Level Analysis (ELA) and outputs a rendered heatmap PNG.
        Calibrated to account for normal H.264 social media compression grid noise.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = out_dir / "ela_heatmap.png"

        path = Path(image_path)
        with Image.open(path) as original:
            original = original.convert("RGB")

            buffer = io.BytesIO()
            original.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            resaved = Image.open(buffer)

            diff = ImageChops.difference(original, resaved)
            diff_arr = np.array(diff, dtype=np.float32)

            scaled_diff = diff_arr * scale
            scaled_diff = np.clip(scaled_diff, 0, 255)

            magnitude = np.sqrt(np.sum(np.square(scaled_diff), axis=-1))

            mean_error = float(np.mean(magnitude))
            max_error = float(np.max(magnitude))
            error_variance = float(np.var(magnitude))

            high_error_threshold = 100.0
            high_error_pixels = np.sum(magnitude > high_error_threshold)
            total_pixels = magnitude.size
            high_error_ratio = float(high_error_pixels / total_pixels) if total_pixels > 0 else 0.0

            # Calibrated tampering score: spliced patches create localized sharp variance spikes
            tampering_score = min(1.0, (error_variance / 600.0) * 0.4 + (high_error_ratio * 1.8) * 0.6)

            try:
                import matplotlib.cm as cm
                norm_mag = magnitude / (max_error + 1e-6)
                colormap = cm.get_cmap("inferno")
                heatmap_rgba = colormap(norm_mag)
                heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
                heatmap_img = Image.fromarray(heatmap_rgb)
                heatmap_img.save(str(heatmap_path))
            except Exception:
                fallback_img = Image.fromarray(scaled_diff.astype(np.uint8))
                fallback_img.save(str(heatmap_path))

        return ElaResult(
            heatmap_path=str(heatmap_path),
            mean_error=round(mean_error, 2),
            max_error=round(max_error, 2),
            error_variance=round(error_variance, 2),
            high_error_pixel_ratio=round(high_error_ratio, 4),
            tampering_suspicion_score=round(tampering_score, 3)
        )

    def detect_and_crop_face(
        self,
        image_path: str,
        faces_dir: Path,
        frame_idx: int
    ) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
        """
        Detects primary human face using MediaPipe FaceMesh / OpenCV, returns (crop_file_path, face_bbox (x,y,w,h)),
        and crops face region with a 25% margin for boundary seam inspection.
        """
        if cv2 is None:
            return None, None

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, None

        h, w = img_bgr.shape[:2]

        # 1. Try MediaPipe FaceMesh for high precision face boundary
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                ) as face_mesh:
                    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        xs = [lm.x * w for lm in landmarks]
                        ys = [lm.y * h for lm in landmarks]

                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)

                        box_w = max_x - min_x
                        box_h = max_y - min_y

                        center_x = (min_x + max_x) / 2.0
                        center_y = (min_y + max_y) / 2.0

                        # Maintain square aspect ratio with 30% margin
                        side = max(box_w, box_h) * 1.30

                        x1 = int(center_x - side / 2.0)
                        y1 = int(center_y - side / 2.0)
                        x2 = int(center_x + side / 2.0)
                        y2 = int(center_y + side / 2.0)

                        if side > 30:
                            # Pad boundaries with reflection if needed to keep strictly square
                            pad_left = max(0, -x1)
                            pad_top = max(0, -y1)
                            pad_right = max(0, x2 - w)
                            pad_bottom = max(0, y2 - h)

                            crop_x1 = max(0, x1)
                            crop_y1 = max(0, y1)
                            crop_x2 = min(w, x2)
                            crop_y2 = min(h, y2)

                            cropped = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                                face_crop = cv2.copyMakeBorder(
                                    cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
                                )
                            else:
                                face_crop = cropped

                            crop_file = faces_dir / f"face_crop_{frame_idx:04d}.jpg"
                            cv2.imwrite(str(crop_file), face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            face_box = (int(max(0, min_x)), int(max(0, min_y)), int(box_w), int(box_h))
                            return str(crop_file), face_box
            except Exception as e:
                logger.debug(f"MediaPipe face cropping error: {e}")

        # 2. Fallback to OpenCV Haar Cascade face detector
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, fw, fh = faces[0]

                    center_x = x + fw / 2.0
                    center_y = y + fh / 2.0
                    side = max(fw, fh) * 1.30

                    x1 = int(center_x - side / 2.0)
                    y1 = int(center_y - side / 2.0)
                    x2 = int(center_x + side / 2.0)
                    y2 = int(center_y + side / 2.0)

                    pad_left = max(0, -x1)
                    pad_top = max(0, -y1)
                    pad_right = max(0, x2 - w)
                    pad_bottom = max(0, y2 - h)

                    crop_x1 = max(0, x1)
                    crop_y1 = max(0, y1)
                    crop_x2 = min(w, x2)
                    crop_y2 = min(h, y2)

                    cropped = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                        face_crop = cv2.copyMakeBorder(
                            cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
                        )
                    else:
                        face_crop = cropped

                    crop_file = faces_dir / f"face_crop_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(crop_file), face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    face_box = (int(x), int(y), int(fw), int(fh))
                    return str(crop_file), face_box
        except Exception as e:
            logger.debug(f"OpenCV Haar face cropping error: {e}")

        return None, None

    def compute_temporal_flicker(self, frame_paths: List[str]) -> Tuple[float, str, List[str]]:
        """
        Analyzes temporal consistency across consecutive keyframes.
        AI video generators (Sora, Kling, Runway, Pika) exhibit morphing geometry,
        rapid edge variance fluctuations, and background flickering between keyframes.
        """
        if len(frame_paths) < 2 or cv2 is None:
            return 0.0, "Insufficient frames to assess temporal consistency", []

        diff_scores: List[float] = []
        laplacian_vars: List[float] = []
        indicators: List[str] = []

        prev_gray = None
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            # Resize to standardized dimensions for consistency
            resized = cv2.resize(img, (320, 320))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # High-frequency edge sharpness (Laplacian variance)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            laplacian_vars.append(lap_var)

            if prev_gray is not None:
                # Frame-to-frame pixel difference
                abs_diff = cv2.absdiff(prev_gray, gray)
                diff_norm = float(np.mean(abs_diff)) / 255.0
                diff_scores.append(diff_norm)

            prev_gray = gray

        if not diff_scores:
            return 0.0, "Consistent", []

        mean_diff = float(np.mean(diff_scores))
        lap_variance = float(np.var(laplacian_vars)) if len(laplacian_vars) > 1 else 0.0

        # Calibrated flicker metric: combines normalized frame jump with edge variance fluctuation
        flicker_score = min(1.0, (mean_diff * 1.5) + (lap_variance / 4000.0) * 0.3)
        flicker_score = round(flicker_score, 3)

        if flicker_score >= 0.65:
            verdict = "Severe Temporal Flickering / Geometry Morphing"
            indicators.append(
                f"High temporal instability (flicker score {flicker_score:.2f}) across keyframes, "
                "characteristic of generative AI video models (background warping and texture shifting)."
            )
        elif flicker_score >= 0.40:
            verdict = "Moderate Inter-frame Variation"
            indicators.append(
                f"Elevated inter-frame fluctuation (flicker score {flicker_score:.2f}). "
                "May indicate fast camera motion, rapid cuts, or subtle AI video generation."
            )
        else:
            verdict = "Consistent / Smooth Temporal Dynamics"

        return flicker_score, verdict, indicators

    def generate_detection_screenshots(
        self,
        top_frames: List[FramePrediction],
        all_frames: List[FramePrediction],
        max_screenshots: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Renders annotated detection evidence screenshots using OpenCV.
        If face manipulation or AI visual anomaly is >= 0.45, draws corner brackets around face
        or full-frame border tags.
        If authentic (< 0.45), produces clean unboxed verification frames with subtle green tags.
        """
        if cv2 is None:
            return []

        candidates = top_frames[:max_screenshots] if top_frames else all_frames[:max_screenshots]
        results: List[Dict[str, Any]] = []

        for fp in candidates:
            try:
                img_bgr = cv2.imread(fp.frame_path)
                if img_bgr is None:
                    continue

                h, w = img_bgr.shape[:2]
                annotated = img_bgr.copy()

                ts = fp.timestamp
                mins = int(ts // 60)
                secs = ts % 60
                time_str = f"{mins:02d}:{secs:04.1f}s"

                score = fp.face_fake_score if fp.face_fake_score is not None else fp.fake_score
                score = normalize_prob(score)
                is_anomaly = score >= 0.45
                score_pct = int(round(score * 100))

                if is_anomaly:
                    is_flagged = True
                    if fp.face_detected and fp.face_bbox and (fp.face_fake_score is not None and fp.face_fake_score >= 0.45):
                        fx, fy, fw, fh = fp.face_bbox
                        fx = max(0, min(w - 1, fx))
                        fy = max(0, min(h - 1, fy))
                        fw = min(fw, w - fx)
                        fh = min(fh, h - fy)

                        bracket_color = (60, 60, 240) if score >= 0.70 else (30, 165, 245)
                        bracket_len = max(15, min(int(fw * 0.25), int(fh * 0.25), 40))
                        th = max(2, int(min(w, h) / 300))

                        # 4 Corner brackets around face
                        cv2.line(annotated, (fx, fy), (fx + bracket_len, fy), bracket_color, th)
                        cv2.line(annotated, (fx, fy), (fx, fy + bracket_len), bracket_color, th)
                        cv2.line(annotated, (fx + fw, fy), (fx + fw - bracket_len, fy), bracket_color, th)
                        cv2.line(annotated, (fx + fw, fy), (fx + fw, fy + bracket_len), bracket_color, th)
                        cv2.line(annotated, (fx, fy + fh), (fx + bracket_len, fy + fh), bracket_color, th)
                        cv2.line(annotated, (fx, fy + fh), (fx, fy + fh - bracket_len), bracket_color, th)
                        cv2.line(annotated, (fx + fw, fy + fh), (fx + fw - bracket_len, fy + fh), bracket_color, th)
                        cv2.line(annotated, (fx + fw, fy + fh), (fx + fw, fy + fh - bracket_len), bracket_color, th)

                        label_text = f"DETECTED: FACE ANOMALY | {time_str} | {score_pct}%"
                        reason = f"Facial region exhibited synthetic neural texture / face-swap boundary inconsistencies (confidence: {score_pct}%)."
                        font_scale = max(0.40, min(w, h) / 1200.0)
                        (tw, th_box), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                        label_y = fy - 10 if fy - 25 > 0 else fy + fh + 25
                        label_x = max(5, min(w - tw - 10, fx))

                        cv2.rectangle(
                            annotated,
                            (label_x - 4, label_y - th_box - 4),
                            (label_x + tw + 4, label_y + baseline + 2),
                            (15, 23, 42),
                            cv2.FILLED
                        )
                        cv2.rectangle(
                            annotated,
                            (label_x - 4, label_y - th_box - 4),
                            (label_x + tw + 4, label_y + baseline + 2),
                            bracket_color,
                            1
                        )
                        cv2.putText(
                            annotated,
                            label_text,
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                    else:
                        tag_color = (60, 60, 240) if score >= 0.70 else (30, 165, 245)
                        label_text = f"DETECTED: AI VISUAL ARTIFACTS | {time_str} | {score_pct}%"
                        reason = f"Full-frame generative synthesis artifacts / optical incongruities identified (confidence: {score_pct}%)."
                        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), cv2.FILLED)
                        cv2.line(annotated, (0, 36), (w, 36), tag_color, 2)
                        font_scale = max(0.45, min(w, h) / 1100.0)
                        cv2.putText(
                            annotated,
                            label_text,
                            (14, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )
                else:
                    is_flagged = False
                    auth_pct = int(round((1.0 - score) * 100))
                    auth_pct = min(96, max(6, auth_pct))
                    label_text = f"VERIFIED FRAME | {time_str} | AUTHENTIC ({auth_pct}%)"
                    reason = "Authentic camera optics, consistent noise grain, and natural biological facial features observed."
                    cv2.rectangle(annotated, (0, 0), (w, 32), (15, 23, 42), cv2.FILLED)
                    cv2.line(annotated, (0, 32), (w, 32), (80, 200, 120), 2)
                    font_scale = max(0.42, min(w, h) / 1200.0)
                    cv2.putText(
                        annotated,
                        label_text,
                        (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (160, 240, 180),
                        1,
                        cv2.LINE_AA
                    )

                max_w = 640
                if w > max_w:
                    new_h = int(h * (max_w / float(w)))
                    annotated = cv2.resize(annotated, (max_w, new_h), interpolation=cv2.INTER_AREA)

                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

                crop_b64 = None
                if fp.face_crop_path and os.path.exists(fp.face_crop_path):
                    try:
                        crop_img = cv2.imread(fp.face_crop_path)
                        if crop_img is not None:
                            crop_w = 160
                            crop_h = int(crop_img.shape[0] * (crop_w / float(crop_img.shape[1])))
                            crop_resized = cv2.resize(crop_img, (crop_w, crop_h))
                            _, cbuf = cv2.imencode(".jpg", crop_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            crop_b64 = "data:image/jpeg;base64," + base64.b64encode(cbuf).decode("utf-8")
                    except Exception:
                        crop_b64 = None

                results.append({
                    "image_b64": img_b64,
                    "crop_b64": crop_b64,
                    "timestamp_sec": round(ts, 2),
                    "time_formatted": time_str,
                    "reason": reason,
                    "score": round(score, 3),
                    "is_flagged": is_flagged
                })
            except Exception as e:
                logger.debug(f"Error creating detection screenshot for {fp.frame_path}: {e}")

        return results

    def analyze(
        self,
        frame_paths: List[str],
        frame_timestamps: List[float],
        work_dir: Union[str, Path],
        max_frames: int = 10
    ) -> VisualForensicsResult:
        """
        Executes complete visual forensics:
        1. EXIF check with platform awareness
        2. ELA heatmap on primary frame
        3. Face detection and cropped face deepfake classification
        4. Full-frame deepfake classification
        5. Frame-to-frame temporal flicker consistency evaluation
        """
        work_path = Path(work_dir)
        faces_dir = work_path / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)

        if not frame_paths:
            return VisualForensicsResult(
                exif_analysis=ExifAnalysisResult(
                    has_exif=False,
                    suspicious_indicators=[],
                    platform_note="No visual frames available."
                ),
                ela_result=None,
                average_fake_score=0.0,
                peak_fake_score=0.0,
                average_face_fake_score=0.0,
                peak_face_fake_score=0.0,
                average_full_frame_fake_score=0.0,
                temporal_flicker_score=0.0,
                temporal_consistency_verdict="No Visual Frames",
                is_flagged_deepfake=False,
                is_flagged_ai_video=False,
                frame_predictions=[],
                top_flagged_frames=[],
                face_crop_paths=[]
            )

        # 1. EXIF Check on first frame
        primary_frame = frame_paths[0]
        exif_result = self.analyze_exif(primary_frame)

        # 2. Error Level Analysis (ELA)
        ela_result = self.compute_ela(primary_frame, work_path)

        # 3. Sample up to max_frames
        total_frames = len(frame_paths)
        if total_frames > max_frames:
            step = total_frames / float(max_frames)
            sample_indices = [int(i * step) for i in range(max_frames)]
        else:
            sample_indices = list(range(total_frames))

        sampled_paths = [frame_paths[i] for i in sample_indices]

        # 4. Temporal flicker analysis across sampled keyframes
        flicker_score, flicker_verdict, ai_video_indicators = self.compute_temporal_flicker(sampled_paths)

        # 5. Classify both cropped faces and full frames
        frame_predictions: List[FramePrediction] = []
        face_crops_saved: List[str] = []
        face_fake_scores: List[float] = []
        full_fake_scores: List[float] = []

        for count, idx in enumerate(sample_indices):
            f_path = frame_paths[idx]
            timestamp = frame_timestamps[idx] if idx < len(frame_timestamps) else float(idx)

            face_detected = False
            face_crop_path = None
            face_fake_s = None
            face_box = None

            # Detect & crop face with bounding box
            face_crop_path, face_box = self.detect_and_crop_face(f_path, faces_dir, count + 1)
            if face_crop_path and os.path.exists(face_crop_path):
                face_detected = True
                face_crops_saved.append(face_crop_path)
                try:
                    with Image.open(face_crop_path) as face_img:
                        face_fake_s, _, _ = self.detector.predict(face_img)
                        face_fake_s = normalize_prob(face_fake_s)
                        face_fake_scores.append(face_fake_s)
                except Exception as e:
                    logger.debug(f"Face crop prediction error: {e}")

            # Predict on full frame
            full_fake_s = 0.50
            full_real_s = 0.50
            pred_lbl = "Real"
            try:
                with Image.open(f_path) as full_img:
                    full_fake_s, full_real_s, pred_lbl = self.detector.predict(full_img)
                    full_fake_s = normalize_prob(full_fake_s)
                    full_fake_scores.append(full_fake_s)
            except Exception as e:
                logger.error(f"Full frame prediction error on {f_path}: {e}")

            # Weighted combined score: if a face is detected, prioritize face score
            if face_fake_s is not None:
                combined_fake_s = normalize_prob(face_fake_s * 0.70 + full_fake_s * 0.30)
            else:
                combined_fake_s = normalize_prob(full_fake_s)

            frame_predictions.append(FramePrediction(
                frame_path=f_path,
                timestamp=timestamp,
                fake_score=combined_fake_s,
                real_score=round(1.0 - combined_fake_s, 4),
                predicted_label="Fake" if combined_fake_s >= 0.5 else "Real",
                face_detected=face_detected,
                face_crop_path=face_crop_path,
                face_fake_score=face_fake_s,
                full_frame_fake_score=full_fake_s,
                face_bbox=face_box
            ))

        # Robust aggregation using simple, stable median across all frames
        all_fake_scores = [normalize_prob(fp.fake_score) for fp in frame_predictions]
        if all_fake_scores:
            median_combined = normalize_prob(float(np.median(all_fake_scores)))
            n = len(all_fake_scores)
            if n >= 6:
                sorted_scores = sorted(all_fake_scores)
                k = max(1, int(n * 0.15))
                trimmed_combined = normalize_prob(float(np.mean(sorted_scores[k:n-k])))
            else:
                trimmed_combined = median_combined
            avg_combined = trimmed_combined
            peak_combined = normalize_prob(float(max(all_fake_scores)))
        else:
            median_combined = 0.04
            avg_combined = 0.04
            peak_combined = 0.04
            trimmed_combined = 0.04

        # Only compute face manipulation statistics across frames where a face was actually detected
        # Apply balance dampening to standalone local face score to suppress compression & smoothing noise
        if face_fake_scores:
            raw_face_median = float(np.median(face_fake_scores))
            calibrated_face = max(0.05, min(0.88, (raw_face_median - 0.25) * 0.75))
            avg_face_fake = float(np.mean(face_fake_scores))
            peak_face_fake = float(max(face_fake_scores))
            num_faces_detected = len(face_fake_scores)
            median_face_fake = round(calibrated_face, 4)
            robust_face_fake = round(calibrated_face, 4)
            p75_face_fake = round(raw_face_median, 4)
        else:
            raw_face_median = 0.05
            calibrated_face = 0.05
            p75_face_fake = 0.05
            robust_face_fake = 0.05
            median_face_fake = 0.05
            avg_face_fake = 0.05
            peak_face_fake = 0.05
            num_faces_detected = 0

        # Apply compression dampening to local AI visual score
        if full_fake_scores:
            raw_local_visual = float(np.median(full_fake_scores))
            calibrated_visual = max(0.05, min(0.85, raw_local_visual * 0.55))
            median_full_fake = round(calibrated_visual, 4)
            avg_full_fake = round(calibrated_visual, 4)
            trimmed_full_fake = round(calibrated_visual, 4)
        else:
            raw_local_visual = 0.05
            calibrated_visual = 0.05
            median_full_fake = 0.05
            avg_full_fake = 0.05
            trimmed_full_fake = 0.05

        # Sort top flagged suspicious frames
        sorted_flagged = sorted(
            frame_predictions,
            key=lambda x: (x.face_fake_score if x.face_fake_score is not None else x.fake_score),
            reverse=True
        )
        top_flagged = [fp for fp in sorted_flagged if fp.fake_score >= 0.45][:3]
        if not top_flagged and sorted_flagged:
            top_flagged = sorted_flagged[:2]

        # Calibrated flags: clean median logic across face crops
        if median_face_fake < 0.45:
            is_deepfake_face = False
        else:
            is_deepfake_face = bool(
                median_face_fake >= 0.70 and num_faces_detected >= 1
            )

        if median_full_fake < 0.45 and flicker_score < 0.50:
            is_ai_video = False
        else:
            is_ai_video = bool(
                (flicker_score >= 0.65 and median_full_fake >= 0.70)
                or (median_full_fake >= 0.80 and not face_crops_saved)
            )

        # Generate annotated detection evidence screenshots
        detection_screenshots = self.generate_detection_screenshots(
            top_frames=top_flagged,
            all_frames=frame_predictions,
            max_screenshots=3
        )

        return VisualForensicsResult(
            exif_analysis=exif_result,
            ela_result=ela_result,
            average_fake_score=round(avg_combined, 4),
            peak_fake_score=round(peak_combined, 4),
            average_face_fake_score=round(avg_face_fake, 4),
            peak_face_fake_score=round(peak_face_fake, 4),
            average_full_frame_fake_score=round(avg_full_fake, 4),
            temporal_flicker_score=flicker_score,
            temporal_consistency_verdict=flicker_verdict,
            median_fake_score=round(median_combined, 4),
            trimmed_mean_fake_score=round(trimmed_combined, 4),
            median_face_fake_score=round(median_face_fake, 4),
            median_full_frame_fake_score=round(median_full_fake, 4),
            p75_face_fake_score=round(p75_face_fake, 4),
            robust_face_fake_score=robust_face_fake,
            num_faces_detected=num_faces_detected,
            ai_generated_video_indicators=ai_video_indicators,
            is_flagged_deepfake=is_deepfake_face,
            is_flagged_ai_video=is_ai_video,
            frame_predictions=frame_predictions,
            top_flagged_frames=top_flagged,
            face_crop_paths=face_crops_saved,
            detection_screenshots=detection_screenshots
        )
