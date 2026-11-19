"""
TruthLens Visual Forensics Module (Optimized for Reels & Social Video)
Handles:
1) EXIF metadata inspection with Instagram H.264 platform-awareness
2) Error Level Analysis (ELA) heatmap generation saved as PNG
3) Face detection & cropping (OpenCV/MediaPipe) with targeted face-swap classification
4) Full-frame deepfake classification (dima806/deepfake_vs_real_image_detection)
5) Frame-to-frame temporal flicker & consistency analysis for AI-generated video detection
"""

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
    ai_generated_video_indicators: List[str] = field(default_factory=list)
    is_flagged_deepfake: bool = False
    is_flagged_ai_video: bool = False
    frame_predictions: List[FramePrediction] = field(default_factory=list)
    top_flagged_frames: List[FramePrediction] = field(default_factory=list)
    face_crop_paths: List[str] = field(default_factory=list)
    model_name: str = "dima806/deepfake_vs_real_image_detection"


class HuggingFaceImageDeepfakeDetector:
    """
    Lazy-loaded classifier using HuggingFace's
    'dima806/deepfake_vs_real_image_detection'.
    """
    _instance = None

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

        id2label = getattr(self.model.config, "id2label", {0: "Real", 1: "Fake"})
        fake_prob = 0.0
        real_prob = 0.0

        for idx, prob in enumerate(probs):
            label = str(id2label.get(idx, f"label_{idx}")).lower()
            if "fake" in label or "manipulated" in label:
                fake_prob = float(prob)
            elif "real" in label or "authentic" in label:
                real_prob = float(prob)

        if fake_prob == 0.0 and real_prob == 0.0 and len(probs) >= 2:
            fake_prob = float(probs[1])
            real_prob = float(probs[0])

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
    ) -> Optional[str]:
        """
        Detects primary human face using MediaPipe FaceMesh / OpenCV and crops face region
        with a 25% margin (to inspect face-swap blending seams around jawline and forehead).
        """
        if cv2 is None:
            return None

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None

        h, w = img_bgr.shape[:2]
        crop_path = None

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

                        # 25% expansion margin for jawline and forehead seams
                        pad_x = box_w * 0.25
                        pad_y = box_h * 0.25

                        x1 = max(0, int(min_x - pad_x))
                        y1 = max(0, int(min_y - pad_y))
                        x2 = min(w, int(max_x + pad_x))
                        y2 = min(h, int(max_y + pad_y))

                        if (x2 - x1) > 30 and (y2 - y1) > 30:
                            face_crop = img_bgr[y1:y2, x1:x2]
                            crop_file = faces_dir / f"face_crop_{frame_idx:04d}.jpg"
                            cv2.imwrite(str(crop_file), face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            return str(crop_file)
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
                    # Select largest face
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, fw, fh = faces[0]

                    pad_x = int(fw * 0.20)
                    pad_y = int(fh * 0.20)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + fw + pad_x)
                    y2 = min(h, y + fh + pad_y)

                    face_crop = img_bgr[y1:y2, x1:x2]
                    crop_file = faces_dir / f"face_crop_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(crop_file), face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    return str(crop_file)
        except Exception as e:
            logger.debug(f"OpenCV Haar face cropping error: {e}")

        return crop_path

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

            # Detect & crop face
            face_crop_path = self.detect_and_crop_face(f_path, faces_dir, count + 1)
            if face_crop_path and os.path.exists(face_crop_path):
                face_detected = True
                face_crops_saved.append(face_crop_path)
                try:
                    with Image.open(face_crop_path) as face_img:
                        face_fake_s, _, _ = self.detector.predict(face_img)
                        face_fake_scores.append(face_fake_s)
                except Exception as e:
                    logger.debug(f"Face crop prediction error: {e}")

            # Predict on full frame
            full_fake_s = 0.5
            full_real_s = 0.5
            pred_lbl = "Real"
            try:
                with Image.open(f_path) as full_img:
                    full_fake_s, full_real_s, pred_lbl = self.detector.predict(full_img)
                    full_fake_scores.append(full_fake_s)
            except Exception as e:
                logger.error(f"Full frame prediction error on {f_path}: {e}")

            # Weighted combined score: if a face is detected, prioritize face score
            if face_fake_s is not None:
                # 70% face crop, 30% full frame
                combined_fake_s = round(face_fake_s * 0.70 + full_fake_s * 0.30, 4)
            else:
                combined_fake_s = full_fake_s

            frame_predictions.append(FramePrediction(
                frame_path=f_path,
                timestamp=timestamp,
                fake_score=combined_fake_s,
                real_score=round(1.0 - combined_fake_s, 4),
                predicted_label="Fake" if combined_fake_s >= 0.5 else "Real",
                face_detected=face_detected,
                face_crop_path=face_crop_path,
                face_fake_score=face_fake_s,
                full_frame_fake_score=full_fake_s
            ))

        # Aggregate metrics
        avg_combined = float(np.mean([fp.fake_score for fp in frame_predictions])) if frame_predictions else 0.0
        peak_combined = float(max([fp.fake_score for fp in frame_predictions])) if frame_predictions else 0.0

        avg_face_fake = float(np.mean(face_fake_scores)) if face_fake_scores else 0.0
        peak_face_fake = float(max(face_fake_scores)) if face_fake_scores else 0.0

        avg_full_fake = float(np.mean(full_fake_scores)) if full_fake_scores else 0.0

        # Sort top flagged frames
        sorted_flagged = sorted(
            frame_predictions,
            key=lambda x: x.fake_score,
            reverse=True
        )
        top_flagged = [fp for fp in sorted_flagged if fp.fake_score > 0.45][:5]
        if not top_flagged and sorted_flagged:
            top_flagged = [sorted_flagged[0]]

        # Determine separate flags
        is_deepfake_face = (peak_face_fake >= 0.65 or avg_face_fake >= 0.55)
        is_ai_video = (flicker_score >= 0.55 or (avg_full_fake >= 0.60 and not face_detected))

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
            ai_generated_video_indicators=ai_video_indicators,
            is_flagged_deepfake=is_deepfake_face,
            is_flagged_ai_video=is_ai_video,
            frame_predictions=frame_predictions,
            top_flagged_frames=top_flagged,
            face_crop_paths=face_crops_saved
        )
