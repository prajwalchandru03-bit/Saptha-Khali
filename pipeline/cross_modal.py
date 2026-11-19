"""
TruthLens Cross-Modal Forensics Module
Handles:
1) Lip-sync check: MediaPipe FaceMesh mouth-opening vs Librosa audio RMS energy correlation
2) CLIP (openai/clip-vit-base-patch32) caption-vs-frame semantic similarity
3) Audio-vs-Visual scene mismatch: CLIP zero-shot visual scene labels vs
   MIT/ast-finetuned-audioset-10-10-0.4593 background audio tags
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import librosa
except ImportError:
    librosa = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger("truthlens.cross_modal")

# Candidate scenes for zero-shot visual classification
CANDIDATE_VISUAL_SCENES = [
    "indoor office or classroom",
    "indoor bedroom or domestic living room",
    "outdoor urban city street with traffic",
    "outdoor natural park, trees, and landscape",
    "beach or ocean water",
    "stadium, arena, or large public crowd rally",
    "nightclub, concert, or music performance",
    "formal TV news studio or press conference",
    "warzone, battlefield, explosions, or military conflict",
    "inside an automobile or vehicle interior"
]


@dataclass
class LipSyncResult:
    """Forensic evaluation of audio-visual lip synchronization."""
    correlation_score: Optional[float]
    faces_detected: bool
    num_frames_with_face: int
    verdict: str
    explanation: str


@dataclass
class ClipCaptionResult:
    """Semantic consistency evaluation between caption and visual content."""
    mean_similarity: float
    max_similarity: float
    min_similarity: float
    alignment_verdict: str
    caption_evaluated: Optional[str]
    frame_scores: List[Dict[str, Union[float, str]]] = field(default_factory=list)


@dataclass
class SceneMismatchResult:
    """Acoustic-visual contextual consistency analysis."""
    top_visual_scenes: List[Dict[str, Union[str, float]]]
    top_audio_tags: List[Dict[str, Union[str, float]]]
    mismatch_detected: bool
    mismatch_score: float  # 0.0 to 1.0
    explanation: str


@dataclass
class CrossModalResult:
    """Unified cross-modal forensics result."""
    lip_sync: LipSyncResult
    caption_alignment: ClipCaptionResult
    scene_consistency: SceneMismatchResult
    is_cross_modal_anomaly_detected: bool


class ClipModelManager:
    """
    Lazy-loaded singleton for CLIP (openai/clip-vit-base-patch32).
    Used for:
    - Caption vs Frame semantic similarity
    - Zero-shot visual scene classification
    """
    _instance = None

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id = "openai/clip-vit-base-patch32"
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
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP model: {self.model_id} onto {self.device}...")
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("CLIP model loaded successfully.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Could not load CLIP model ({self.model_id}): {e}. Fallback scoring will be used.")

    def compute_caption_similarity(self, caption: str, images: List[Image.Image]) -> List[float]:
        """Calculates cosine similarity between caption text and each image."""
        if not self._is_loaded and self._load_error is None:
            self.load_model()

        if self._load_error is not None or self.model is None or self.processor is None:
            return [0.5] * len(images)

        import torch
        rgb_images = [img.convert("RGB") for img in images]
        inputs = self.processor(
            text=[caption],
            images=rgb_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Normalize embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarities = (image_embeds @ text_embeds.T).squeeze(-1).tolist()

        if isinstance(similarities, float):
            similarities = [similarities]

        return [round(float(s), 4) for s in similarities]

    def classify_visual_scene(self, images: List[Image.Image], candidate_labels: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Zero-shot visual scene classification over candidate scene descriptions."""
        if not self._is_loaded and self._load_error is None:
            self.load_model()

        if self._load_error is not None or self.model is None or self.processor is None:
            return [{"label": candidate_labels[0], "score": 0.5}]

        import torch
        rgb_images = [img.convert("RGB") for img in images[:5]]  # sample up to 5
        inputs = self.processor(
            text=candidate_labels,
            images=rgb_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # (num_images, num_labels)
            probs = torch.nn.functional.softmax(logits_per_image, dim=-1)
            mean_probs = probs.mean(dim=0).tolist()

        results = [
            {"label": lbl, "score": round(float(p), 4)}
            for lbl, p in zip(candidate_labels, mean_probs)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:3]


class AstAudioClassifier:
    """
    Lazy-loaded singleton for Audio Spectrogram Transformer (AST)
    'MIT/ast-finetuned-audioset-10-10-0.4593' to detect background acoustic events.
    """
    _instance = None

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.feature_extractor = None
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
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            logger.info(f"Loading AST audio classifier: {self.model_id} onto {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("AST audio classifier loaded successfully.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Could not load AST audio classifier ({self.model_id}): {e}. Fallback scoring will be used.")

    def classify_audio(self, audio_path: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """Extracts top AudioSet acoustic event tags with probabilities."""
        if not self._is_loaded and self._load_error is None:
            self.load_model()

        if self._load_error is not None or self.model is None or self.feature_extractor is None:
            return [{"tag": "Speech", "score": 0.5}]

        try:
            import soundfile as sf
            import torch

            signal, sr = sf.read(audio_path, dtype="float32")
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)

            # Sample first 10.24 seconds for AST (or full length if shorter)
            max_len = int(10.24 * 16000)
            if len(signal) > max_len:
                signal = signal[:max_len]

            inputs = self.feature_extractor(signal, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).squeeze().tolist()

            id2label = self.model.config.id2label
            indexed_scores = [(id2label.get(i, f"tag_{i}"), float(p)) for i, p in enumerate(probs)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {"tag": tag, "score": round(score, 4)}
                for tag, score in indexed_scores[:top_k]
            ]
        except Exception as e:
            logger.error(f"AST classification failure on {audio_path}: {e}")
            return [{"tag": "General Audio", "score": 0.5}]


class CrossModalAnalyzer:
    """
    Coordinates:
    1) Lip-sync correlation (MediaPipe FaceMesh mouth opening vs Librosa RMS)
    2) CLIP caption vs visual frame similarity
    3) Audio-visual scene consistency (CLIP zero-shot scene vs AST audio tags)
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.clip_mgr = ClipModelManager.get_instance(device=device)
        self.ast_classifier = AstAudioClassifier.get_instance(device=device)

    def analyze_lip_sync(
        self,
        frame_paths: List[str],
        frame_timestamps: List[float],
        audio_path: Optional[str]
    ) -> LipSyncResult:
        """
        Calculates Pearson correlation between MediaPipe FaceMesh mouth opening
        and Librosa audio RMS energy across timestamps.
        """
        if not audio_path or not Path(audio_path).exists():
            return LipSyncResult(
                correlation_score=None,
                faces_detected=False,
                num_frames_with_face=0,
                verdict="No Audio Track",
                explanation="Audio track is absent; cannot compute acoustic-lip synchronization."
            )

        if not frame_paths:
            return LipSyncResult(
                correlation_score=None,
                faces_detected=False,
                num_frames_with_face=0,
                verdict="No Video Frames",
                explanation="Visual frames are absent; lip synchronization not applicable."
            )

        # 1. Compute Audio RMS energy series
        audio_rms_values: List[float] = []
        mouth_openings: List[float] = []
        valid_timestamps: List[float] = []

        try:
            import soundfile as sf
            audio_data, sr = sf.read(audio_path, dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
        except Exception:
            if librosa is not None:
                audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            else:
                audio_data = None
                sr = 16000

        # 2. Extract mouth opening with MediaPipe FaceMesh
        faces_detected_count = 0
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                ) as face_mesh:
                    for idx, f_path in enumerate(frame_paths[:30]):
                        t = frame_timestamps[idx] if idx < len(frame_timestamps) else float(idx)
                        try:
                            import cv2
                            img = cv2.imread(f_path)
                            if img is None:
                                continue
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            results = face_mesh.process(rgb)

                            if results.multi_face_landmarks:
                                landmarks = results.multi_face_landmarks[0].landmark
                                # Upper lip inner landmark: 13, lower lip inner: 14
                                # Forehead: 10, chin: 152
                                lip_top = landmarks[13]
                                lip_bottom = landmarks[14]
                                face_top = landmarks[10]
                                face_bottom = landmarks[152]

                                lip_dist = np.sqrt((lip_top.x - lip_bottom.x)**2 + (lip_top.y - lip_bottom.y)**2)
                                face_height = max(1e-4, np.sqrt((face_top.x - face_bottom.x)**2 + (face_top.y - face_bottom.y)**2))
                                norm_opening = float(lip_dist / face_height)

                                # Extract audio RMS energy around timestamp [t - 0.25, t + 0.25]
                                if audio_data is not None and len(audio_data) > 0:
                                    start_sample = max(0, int((t - 0.25) * sr))
                                    end_sample = min(len(audio_data), int((t + 0.25) * sr))
                                    if end_sample > start_sample:
                                        chunk = audio_data[start_sample:end_sample]
                                        rms_val = float(np.sqrt(np.mean(chunk**2)))
                                    else:
                                        rms_val = 0.0
                                else:
                                    rms_val = 0.0

                                mouth_openings.append(norm_opening)
                                audio_rms_values.append(rms_val)
                                valid_timestamps.append(t)
                                faces_detected_count += 1
                        except Exception as e:
                            logger.debug(f"FaceMesh frame processing error: {e}")
            except Exception as e:
                logger.warning(f"MediaPipe FaceMesh initialization failed: {e}")

        # Evaluate correlation
        if faces_detected_count < 3 or len(mouth_openings) < 3:
            return LipSyncResult(
                correlation_score=None,
                faces_detected=(faces_detected_count > 0),
                num_frames_with_face=faces_detected_count,
                verdict="Insufficient Face Samples",
                explanation="No clear human face or mouth landmarks detected across enough frames to calculate correlation."
            )

        # Pearson correlation
        m_arr = np.array(mouth_openings)
        a_arr = np.array(audio_rms_values)

        if np.std(m_arr) < 1e-6 or np.std(a_arr) < 1e-6:
            r = 0.0
        else:
            r = float(np.corrcoef(m_arr, a_arr)[0, 1])
            if np.isnan(r):
                r = 0.0

        r_round = round(r, 3)
        if r >= 0.35:
            verdict = "Synchronized (Natural Lip-Speech Dynamics)"
            explanation = f"Lip opening correlates well with acoustic energy bursts (r = {r_round}), characteristic of genuine speech."
        elif 0.10 <= r < 0.35:
            verdict = "Weak / Borderline Synchronization"
            explanation = f"Lip movement shows marginal correlation with audio bursts (r = {r_round}). Could indicate slight dubbing, fast cuts, or subtle manipulation."
        else:
            verdict = "Severe Desynchronization (Potential Talking-Head / Audio Replacement)"
            explanation = f"Lip dynamics do not match audio speech energy (r = {r_round}). Highly indicative of synthetic dubbing, voice replacement, or still-frame puppet animation."

        return LipSyncResult(
            correlation_score=r_round,
            faces_detected=True,
            num_frames_with_face=faces_detected_count,
            verdict=verdict,
            explanation=explanation
        )

    def analyze_caption_similarity(
        self,
        caption: Optional[str],
        frame_paths: List[str],
        frame_timestamps: List[float]
    ) -> ClipCaptionResult:
        """
        Uses CLIP to measure semantic similarity between caption and video frames.
        Flags context manipulation and out-of-context miscaptioning.
        """
        if not caption or not caption.strip():
            return ClipCaptionResult(
                mean_similarity=0.0,
                max_similarity=0.0,
                min_similarity=0.0,
                alignment_verdict="No Caption Provided",
                caption_evaluated=None,
                frame_scores=[]
            )

        if not frame_paths:
            return ClipCaptionResult(
                mean_similarity=0.0,
                max_similarity=0.0,
                min_similarity=0.0,
                alignment_verdict="No Visual Content",
                caption_evaluated=caption,
                frame_scores=[]
            )

        # Sample up to 8 frames evenly
        total = len(frame_paths)
        sample_indices = [int(i * (total / min(total, 8))) for i in range(min(total, 8))]

        images: List[Image.Image] = []
        sampled_ts: List[float] = []

        for idx in sample_indices:
            p = frame_paths[idx]
            ts = frame_timestamps[idx] if idx < len(frame_timestamps) else float(idx)
            try:
                with Image.open(p) as img:
                    images.append(img.copy())
                    sampled_ts.append(ts)
            except Exception:
                continue

        if not images:
            return ClipCaptionResult(
                mean_similarity=0.0,
                max_similarity=0.0,
                min_similarity=0.0,
                alignment_verdict="Image Read Error",
                caption_evaluated=caption,
                frame_scores=[]
            )

        sims = self.clip_mgr.compute_caption_similarity(caption, images)
        frame_scores = [
            {"timestamp": str(ts), "similarity": s}
            for ts, s in zip(sampled_ts, sims)
        ]

        mean_s = round(float(np.mean(sims)), 4)
        max_s = round(float(np.max(sims)), 4)
        min_s = round(float(np.min(sims)), 4)

        # In CLIP ViT-B/32, cosine similarities typically range:
        # > 0.25: Strong semantic correlation
        # 0.18 - 0.25: Moderate correlation
        # < 0.18: Divergent / Out of context
        if max_s >= 0.25:
            verdict = "Consistent Semantic Alignment"
        elif max_s >= 0.18:
            verdict = "Moderate Semantic Alignment"
        else:
            verdict = "Context Mismatch / False Caption Discrepancy"

        return ClipCaptionResult(
            mean_similarity=mean_s,
            max_similarity=max_s,
            min_similarity=min_s,
            alignment_verdict=verdict,
            caption_evaluated=caption,
            frame_scores=frame_scores
        )

    def analyze_scene_consistency(
        self,
        frame_paths: List[str],
        audio_path: Optional[str]
    ) -> SceneMismatchResult:
        """
        Cross-evaluates visual scene classification (CLIP zero-shot)
        against acoustic event classification (AST AudioSet tags).
        """
        # 1. Zero-shot visual classification
        visual_scenes: List[Dict[str, Union[str, float]]] = []
        if frame_paths:
            sample_imgs = []
            for p in frame_paths[:5]:
                try:
                    with Image.open(p) as img:
                        sample_imgs.append(img.copy())
                except Exception:
                    pass
            if sample_imgs:
                visual_scenes = self.clip_mgr.classify_visual_scene(sample_imgs, CANDIDATE_VISUAL_SCENES)

        # 2. Audio background classification with AST
        audio_tags: List[Dict[str, Union[str, float]]] = []
        if audio_path and Path(audio_path).exists():
            audio_tags = self.ast_classifier.classify_audio(audio_path, top_k=5)

        if not visual_scenes or not audio_tags:
            return SceneMismatchResult(
                top_visual_scenes=visual_scenes,
                top_audio_tags=audio_tags,
                mismatch_detected=False,
                mismatch_score=0.0,
                explanation="Insufficient audio or visual scene data to test cross-modal environmental consistency."
            )

        # 3. Detect known environmental contradictions
        top_visual = visual_scenes[0]["label"].lower()
        top_audio_names = [t["tag"].lower() for t in audio_tags]

        mismatch_detected = False
        mismatch_score = 0.0
        reasons: List[str] = []

        # Contradiction rule sets:
        # A) Indoor quiet scene with extreme outdoor combat/traffic audio
        if ("indoor" in top_visual or "bedroom" in top_visual or "office" in top_visual):
            combat_tags = ["explosion", "gunshot", "siren", "artillery fire", "battle"]
            for ct in combat_tags:
                if any(ct in a for a in top_audio_names):
                    mismatch_detected = True
                    mismatch_score = 0.85
                    reasons.append(f"Visual shows serene indoor space ({top_visual}) while audio contains high-intensity acoustic events ('{ct}').")

        # B) Water / ocean scene with mechanical office / domestic noises
        if ("ocean" in top_visual or "beach" in top_visual or "underwater" in top_visual):
            indoor_tags = ["typing", "telephone bell", "office", "domestic sounds"]
            for it in indoor_tags:
                if any(it in a for a in top_audio_names):
                    mismatch_detected = True
                    mismatch_score = 0.75
                    reasons.append(f"Visual shows marine/beach setting while audio contains indoor artifacts ('{it}').")

        # C) Warzone / explosion visuals with tranquil domestic or cheerful music
        if ("warzone" in top_visual or "conflict" in top_visual):
            cheerful_tags = ["laughter", "giggle", "cheering", "applause", "bird vocalization"]
            for cht in cheerful_tags:
                if any(cht in a for a in top_audio_names):
                    mismatch_detected = True
                    mismatch_score = 0.80
                    reasons.append(f"Visual suggests conflict/crisis while audio exhibits contradictory cheerful acoustics ('{cht}').")

        if mismatch_detected:
            explanation = " ".join(reasons)
        else:
            explanation = f"Visual context ('{visual_scenes[0]['label']}') and acoustic atmosphere ({', '.join(t['tag'] for t in audio_tags[:2])}) appear mutually consistent."

        return SceneMismatchResult(
            top_visual_scenes=visual_scenes,
            top_audio_tags=audio_tags,
            mismatch_detected=mismatch_detected,
            mismatch_score=mismatch_score,
            explanation=explanation
        )

    def analyze(
        self,
        frame_paths: List[str],
        frame_timestamps: List[float],
        audio_path: Optional[str],
        caption: Optional[str] = None
    ) -> CrossModalResult:
        """
        Executes unified cross-modal forensic analysis.
        """
        lip_sync_res = self.analyze_lip_sync(frame_paths, frame_timestamps, audio_path)
        clip_caption_res = self.analyze_caption_similarity(caption, frame_paths, frame_timestamps)
        scene_res = self.analyze_scene_consistency(frame_paths, audio_path)

        anomaly_detected = (
            (lip_sync_res.correlation_score is not None and lip_sync_res.correlation_score < 0.10 and lip_sync_res.faces_detected) or
            ("Mismatch" in clip_caption_res.alignment_verdict) or
            scene_res.mismatch_detected
        )

        return CrossModalResult(
            lip_sync=lip_sync_res,
            caption_alignment=clip_caption_res,
            scene_consistency=scene_res,
            is_cross_modal_anomaly_detected=anomaly_detected
        )
