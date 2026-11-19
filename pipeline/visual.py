"""
TruthLens Visual Forensics Module
Handles:
1) EXIF metadata inspection & anomaly detection
2) Error Level Analysis (ELA) with heatmap generation saved as PNG
3) HuggingFace ViT Deepfake Image Detector (dima806/deepfake_vs_real_image_detection)
"""

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ExifTags

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
    """Per-frame classification output."""
    frame_path: str
    timestamp: float
    fake_score: float
    real_score: float
    predicted_label: str


@dataclass
class VisualForensicsResult:
    """Comprehensive visual forensics results across frames."""
    exif_analysis: ExifAnalysisResult
    ela_result: Optional[ElaResult]
    average_fake_score: float
    peak_fake_score: float
    is_flagged_deepfake: bool
    frame_predictions: List[FramePrediction] = field(default_factory=list)
    top_flagged_frames: List[FramePrediction] = field(default_factory=list)
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
            # Fallback heuristic if weights cannot be downloaded (offline/mock)
            return 0.5, 0.5, "Inconclusive (Model Offline)"

        import torch
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        # Label mapping in dima806/deepfake_vs_real_image_detection:
        # id2label usually: {0: 'Real', 1: 'Fake'} or {0: 'Fake', 1: 'Real'}
        id2label = self.model.config.id2label
        fake_prob = 0.0
        real_prob = 0.0

        for idx, prob in enumerate(probs):
            label = str(id2label.get(idx, f"label_{idx}")).lower()
            if "fake" in label or "manipulated" in label:
                fake_prob = float(prob)
            elif "real" in label or "authentic" in label:
                real_prob = float(prob)

        predicted_label = "Fake" if fake_prob > real_prob else "Real"
        return fake_prob, real_prob, predicted_label


class VisualAnalyzer:
    """Orchestrates EXIF check, ELA analysis, and ViT deepfake classification."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector = HuggingFaceImageDeepfakeDetector.get_instance(device=device)

    def analyze_exif(self, image_path: Union[str, Path]) -> ExifAnalysisResult:
        """Inspects image EXIF metadata for editing software or missing camera provenance."""
        path = Path(image_path)
        suspicious_indicators: List[str] = []
        raw_exif_dict: Dict[str, str] = {}
        software: Optional[str] = None
        make: Optional[str] = None
        model: Optional[str] = None
        datetime_original: Optional[str] = None
        has_exif = False

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
                                    suspicious_indicators.append(f"Software signature matched generative/editing tool: '{val_str}'")
                        elif "make" in tag_lower:
                            make = val_str
                        elif "model" in tag_lower:
                            model = val_str
                        elif "datetimeoriginal" in tag_lower:
                            datetime_original = val_str
        except Exception as e:
            logger.debug(f"EXIF extraction error on {image_path}: {e}")

        if not has_exif:
            suspicious_indicators.append("EXIF metadata is completely absent (commonly stripped by social media recompression or synthetic generation)")

        return ExifAnalysisResult(
            has_exif=has_exif,
            software=software,
            camera_make=make,
            camera_model=model,
            datetime_original=datetime_original,
            suspicious_indicators=suspicious_indicators,
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
        Computes variance, mean error, and pixel tampering suspicion score.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = out_dir / "ela_heatmap.png"

        path = Path(image_path)
        with Image.open(path) as original:
            original = original.convert("RGB")
            
            # Save temporary recompressed JPEG in memory
            buffer = io.BytesIO()
            original.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            resaved = Image.open(buffer)

            # Compute pixel-by-pixel difference
            diff = ImageChops.difference(original, resaved)
            diff_arr = np.array(diff, dtype=np.float32)

            # Amplify difference by scale factor
            scaled_diff = diff_arr * scale
            scaled_diff = np.clip(scaled_diff, 0, 255)

            # Intensity magnitude per pixel: sqrt(R^2 + G^2 + B^2)
            magnitude = np.sqrt(np.sum(np.square(scaled_diff), axis=-1))

            mean_error = float(np.mean(magnitude))
            max_error = float(np.max(magnitude))
            error_variance = float(np.var(magnitude))

            # Quantify high-discrepancy pixels (> 75% of max or above 120 intensity)
            high_error_threshold = 100.0
            high_error_pixels = np.sum(magnitude > high_error_threshold)
            total_pixels = magnitude.size
            high_error_ratio = float(high_error_pixels / total_pixels) if total_pixels > 0 else 0.0

            # Tampering suspicion score (0.0 to 1.0)
            # Spliced/edited regions produce localized high variance in ELA response
            tampering_score = min(1.0, (error_variance / 500.0) * 0.4 + (high_error_ratio * 2.0) * 0.6)

            # Generate Heatmap Colormap using matplotlib or numpy
            try:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm

                norm_mag = magnitude / (max_error + 1e-6)
                colormap = cm.get_cmap("inferno")
                heatmap_rgba = colormap(norm_mag)
                heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
                heatmap_img = Image.fromarray(heatmap_rgb)
                heatmap_img.save(str(heatmap_path))
            except Exception:
                # Fallback grayscale amplified diff
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

    def analyze(
        self,
        frame_paths: List[str],
        frame_timestamps: List[float],
        work_dir: Union[str, Path],
        max_frames: int = 30
    ) -> VisualForensicsResult:
        """
        Executes complete visual forensics across extracted frames.
        1. EXIF check on primary frame / source image.
        2. ELA heatmap on representative frame.
        3. Deepfake ViT classification across sampled frames.
        """
        work_path = Path(work_dir)
        if not frame_paths:
            # Handle pure audio edge case
            return VisualForensicsResult(
                exif_analysis=ExifAnalysisResult(has_exif=False, suspicious_indicators=["No visual frames available"]),
                ela_result=None,
                average_fake_score=0.0,
                peak_fake_score=0.0,
                is_flagged_deepfake=False,
                frame_predictions=[],
                top_flagged_frames=[]
            )

        # 1. EXIF Check on first frame or source image
        primary_frame = frame_paths[0]
        exif_result = self.analyze_exif(primary_frame)

        # 2. Error Level Analysis (ELA) on first / primary frame
        ela_result = self.compute_ela(primary_frame, work_path)

        # 3. Deepfake ViT detection across sampled frames
        # Evenly sample up to max_frames to preserve compute efficiency
        total_frames = len(frame_paths)
        if total_frames > max_frames:
            step = total_frames / max_frames
            sample_indices = [int(i * step) for i in range(max_frames)]
        else:
            sample_indices = list(range(total_frames))

        frame_predictions: List[FramePrediction] = []
        for idx in sample_indices:
            f_path = frame_paths[idx]
            timestamp = frame_timestamps[idx] if idx < len(frame_timestamps) else float(idx)
            try:
                with Image.open(f_path) as img:
                    fake_s, real_s, lbl = self.detector.predict(img)
                    frame_predictions.append(FramePrediction(
                        frame_path=f_path,
                        timestamp=timestamp,
                        fake_score=round(fake_s, 4),
                        real_score=round(real_s, 4),
                        predicted_label=lbl
                    ))
            except Exception as e:
                logger.error(f"Error predicting on frame {f_path}: {e}")

        # Compute aggregate scores
        if frame_predictions:
            avg_fake = sum(fp.fake_score for fp in frame_predictions) / len(frame_predictions)
            peak_fake = max(fp.fake_score for fp in frame_predictions)
            sorted_flagged = sorted(
                [fp for fp in frame_predictions if fp.fake_score > 0.45],
                key=lambda x: x.fake_score,
                reverse=True
            )
            top_flagged = sorted_flagged[:5]
        else:
            avg_fake = 0.0
            peak_fake = 0.0
            top_flagged = []

        is_flagged = avg_fake >= 0.5 or peak_fake >= 0.7 or (ela_result and ela_result.tampering_suspicion_score > 0.7)

        return VisualForensicsResult(
            exif_analysis=exif_result,
            ela_result=ela_result,
            average_fake_score=round(avg_fake, 4),
            peak_fake_score=round(peak_fake, 4),
            is_flagged_deepfake=is_flagged,
            frame_predictions=frame_predictions,
            top_flagged_frames=top_flagged
        )
