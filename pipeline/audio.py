"""
TruthLens Audio Forensics Module
Handles:
1) 4-second audio window chunking at 16kHz mono
2) Synthetic Voice Clone / Deepfake Audio Detection (HyperMoon/wav2vec2-base-960h-finetuned-deepfake)
3) Whisper 'base' model timestamped speech transcription
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger("truthlens.audio")


@dataclass
class AudioChunkResult:
    """Forensic result for a 4-second audio slice."""
    chunk_index: int
    start_seconds: float
    end_seconds: float
    fake_score: float
    real_score: float
    predicted_label: str


@dataclass
class TranscriptSegment:
    """Timestamped segment from Whisper."""
    start: float
    end: float
    text: str


@dataclass
class AudioForensicsResult:
    """Comprehensive audio forensics and speech analysis result."""
    has_audio: bool
    duration_seconds: float
    average_voice_clone_score: float
    peak_voice_clone_score: float
    is_flagged_voice_clone: bool
    detected_language: Optional[str]
    full_transcript: str
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    chunk_results: List[AudioChunkResult] = field(default_factory=list)
    top_flagged_chunks: List[AudioChunkResult] = field(default_factory=list)
    model_name: str = "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"


class Wav2Vec2DeepfakeDetector:
    """
    Lazy-loaded classifier using HuggingFace's
    'HyperMoon/wav2vec2-base-960h-finetuned-deepfake'.
    """
    _instance = None

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_id = "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"
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

            logger.info(f"Loading audio deepfake model: {self.model_id} onto {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("Audio deepfake model loaded successfully.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Failed to load audio deepfake detector ({self.model_id}): {e}. Fallback scoring will be used.")

    def predict_chunk(self, audio_data: np.ndarray, sampling_rate: int = 16000) -> Tuple[float, float, str]:
        """
        Runs voice clone detection on a single audio chunk.
        Returns (fake_score, real_score, label).
        """
        if not self._is_loaded and self._load_error is None:
            self.load_model()

        if self._load_error is not None or self.model is None or self.feature_extractor is None:
            return 0.5, 0.5, "Inconclusive (Model Offline)"

        import torch

        # Ensure audio is 1D float32
        audio_arr = np.squeeze(audio_data).astype(np.float32)

        inputs = self.feature_extractor(
            audio_arr,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [1.0 - probs, probs]

        id2label = getattr(self.model.config, "id2label", {0: "real", 1: "fake"})
        fake_prob = 0.0
        real_prob = 0.0

        for idx, prob in enumerate(probs):
            lbl = str(id2label.get(idx, f"label_{idx}")).lower()
            if "fake" in lbl or "spoof" in lbl or "synthetic" in lbl or "clone" in lbl:
                fake_prob = float(prob)
            elif "real" in lbl or "bonafide" in lbl or "genuine" in lbl:
                real_prob = float(prob)

        # In case labels are not named 'fake'/'real', default 1 is fake, 0 is real
        if fake_prob == 0.0 and real_prob == 0.0 and len(probs) >= 2:
            fake_prob = float(probs[1])
            real_prob = float(probs[0])

        predicted_label = "Deepfake / Voice Clone" if fake_prob > real_prob else "Authentic Voice"
        return fake_prob, real_prob, predicted_label


class WhisperTranscriber:
    """
    Lazy-loaded Whisper 'base' transcription model
    with timestamped segment generation.
    """
    _instance = None

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._is_loaded = False
        self._load_error = None
        self._backend = None  # "openai-whisper" or "transformers"

    @classmethod
    def get_instance(cls, device: str = "cpu"):
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def load_model(self):
        if self._is_loaded:
            return
        
        # Try native openai-whisper
        try:
            import whisper
            logger.info("Loading native openai-whisper 'base' model...")
            self.model = whisper.load_model("base", device=self.device)
            self._backend = "openai-whisper"
            self._is_loaded = True
            logger.info("OpenAI Whisper 'base' model loaded.")
            return
        except Exception as e:
            logger.debug(f"Native whisper import/load failed: {e}. Trying transformers pipeline...")

        # Fallback to transformers Whisper pipeline
        try:
            from transformers import pipeline
            logger.info("Loading transformers openai/whisper-base pipeline...")
            self.model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=0 if self.device == "cuda" else -1,
                return_timestamps=True
            )
            self._backend = "transformers"
            self._is_loaded = True
            logger.info("Transformers Whisper 'base' pipeline loaded.")
            return
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Could not load Whisper model: {e}")

    def transcribe(self, audio_path: str) -> Tuple[str, Optional[str], List[TranscriptSegment]]:
        """
        Transcribes audio file to full text with timestamped segments.
        Returns (full_transcript, detected_language, segments).
        """
        if not self._is_loaded and self._load_error is None:
            self.load_model()

        if not self._is_loaded or self.model is None:
            return "Transcript unavailable (Whisper model offline)", None, []

        try:
            if self._backend == "openai-whisper":
                result = self.model.transcribe(audio_path, verbose=False)
                full_text = result.get("text", "").strip()
                lang = result.get("language")
                raw_segments = result.get("segments", [])
                segments = [
                    TranscriptSegment(
                        start=round(float(s["start"]), 2),
                        end=round(float(s["end"]), 2),
                        text=s["text"].strip()
                    )
                    for s in raw_segments
                ]
                return full_text, lang, segments

            elif self._backend == "transformers":
                result = self.model(audio_path)
                full_text = result.get("text", "").strip()
                raw_chunks = result.get("chunks", [])
                segments = [
                    TranscriptSegment(
                        start=round(float(c["timestamp"][0] or 0.0), 2),
                        end=round(float(c["timestamp"][1] or 0.0), 2),
                        text=c["text"].strip()
                    )
                    for c in raw_chunks
                ]
                return full_text, None, segments

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return f"Transcription error: {str(e)}", None, []

        return "", None, []


class AudioAnalyzer:
    """
    Orchestrates 4-second chunking, voice clone detection,
    and Whisper timestamped speech transcription.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector = Wav2Vec2DeepfakeDetector.get_instance(device=device)
        self.transcriber = WhisperTranscriber.get_instance(device=device)

    def load_audio_signal(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
        """Loads audio track as 16kHz mono float32 numpy array."""
        if sf is not None:
            try:
                data, sr = sf.read(audio_path, dtype="float32")
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sr != target_sr and librosa is not None:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                duration = len(data) / target_sr
                return data, duration
            except Exception as e:
                logger.debug(f"soundfile read failed: {e}")

        if librosa is not None:
            data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            duration = len(data) / target_sr
            return data, duration

        raise RuntimeError("Neither soundfile nor librosa is available to read audio.")

    def chunk_audio(
        self,
        audio_data: np.ndarray,
        sr: int = 16000,
        chunk_duration_sec: float = 4.0
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Splits audio signal into 4.0-second non-overlapping windows.
        Returns list of (chunk_samples, start_time, end_time).
        """
        chunk_samples = int(chunk_duration_sec * sr)
        total_samples = len(audio_data)

        if total_samples == 0:
            return []

        chunks: List[Tuple[np.ndarray, float, float]] = []
        num_chunks = int(np.ceil(total_samples / chunk_samples))

        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, total_samples)
            slice_data = audio_data[start_idx:end_idx]

            # Pad short chunks with zeros to maintain 4-sec dimension
            if len(slice_data) < chunk_samples:
                padded = np.zeros(chunk_samples, dtype=np.float32)
                padded[:len(slice_data)] = slice_data
                slice_data = padded

            start_t = round(start_idx / sr, 2)
            end_t = round(end_idx / sr, 2)
            chunks.append((slice_data, start_t, end_t))

        return chunks

    def analyze(self, audio_path: Optional[str]) -> AudioForensicsResult:
        """
        Executes complete audio forensics:
        1. 4-second chunking
        2. Wav2Vec2 voice clone detection per chunk
        3. Whisper 'base' timestamped transcription
        """
        if not audio_path or not Path(audio_path).exists():
            return AudioForensicsResult(
                has_audio=False,
                duration_seconds=0.0,
                average_voice_clone_score=0.0,
                peak_voice_clone_score=0.0,
                is_flagged_voice_clone=False,
                detected_language=None,
                full_transcript="No audio stream detected in media file.",
                transcript_segments=[],
                chunk_results=[],
                top_flagged_chunks=[]
            )

        # 1. Load audio signal
        try:
            signal, duration = self.load_audio_signal(audio_path, target_sr=16000)
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return AudioForensicsResult(
                has_audio=False,
                duration_seconds=0.0,
                average_voice_clone_score=0.0,
                peak_voice_clone_score=0.0,
                is_flagged_voice_clone=False,
                detected_language=None,
                full_transcript=f"Audio decoding failure: {e}",
                transcript_segments=[],
                chunk_results=[],
                top_flagged_chunks=[]
            )

        # 2. Slice into 4-sec audio chunks and run Wav2Vec2 detector
        chunks = self.chunk_audio(signal, sr=16000, chunk_duration_sec=4.0)
        chunk_results: List[AudioChunkResult] = []

        for idx, (chunk_data, start_sec, end_sec) in enumerate(chunks):
            fake_s, real_s, label = self.detector.predict_chunk(chunk_data, sampling_rate=16000)
            chunk_results.append(AudioChunkResult(
                chunk_index=idx,
                start_seconds=start_sec,
                end_seconds=end_sec,
                fake_score=round(fake_s, 4),
                real_score=round(real_s, 4),
                predicted_label=label
            ))

        if chunk_results:
            avg_fake = sum(c.fake_score for c in chunk_results) / len(chunk_results)
            peak_fake = max(c.fake_score for c in chunk_results)
            sorted_flagged = sorted(
                [c for c in chunk_results if c.fake_score > 0.45],
                key=lambda x: x.fake_score,
                reverse=True
            )
            top_flagged = sorted_flagged[:5]
        else:
            avg_fake = 0.0
            peak_fake = 0.0
            top_flagged = []

        is_flagged = avg_fake >= 0.5 or peak_fake >= 0.75

        # 3. Whisper 'base' timestamped transcription
        full_transcript, language, segments = self.transcriber.transcribe(audio_path)

        return AudioForensicsResult(
            has_audio=True,
            duration_seconds=round(duration, 2),
            average_voice_clone_score=round(avg_fake, 4),
            peak_voice_clone_score=round(peak_fake, 4),
            is_flagged_voice_clone=is_flagged,
            detected_language=language,
            full_transcript=full_transcript,
            transcript_segments=segments,
            chunk_results=chunk_results,
            top_flagged_chunks=top_flagged
        )
