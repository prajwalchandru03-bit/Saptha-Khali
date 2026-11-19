"""
TruthLens Ingestion Module
Handles:
- Social media URL downloads (YouTube, Instagram, TikTok, etc.) via yt-dlp
- Direct media file uploads (Video, Image, Audio)
- SHA-256 cryptographic provenance hashing
- FFmpeg splitting: 1 fps visual frames + 16kHz mono WAV audio extraction
"""

import hashlib
import logging
import mimetypes
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

logger = logging.getLogger("truthlens.ingest")


@dataclass
class IngestResult:
    """Forensic ingestion output and metadata container."""
    sha256_hash: str
    media_type: str  # "video", "image", "audio", "unknown"
    original_filename: str
    original_filepath: str
    work_dir: str
    frames_dir: str
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict = field(default_factory=dict)


def compute_sha256(filepath_or_bytes: Union[str, Path, bytes]) -> str:
    """Computes SHA-256 hash for forensic chain of custody tracking."""
    hasher = hashlib.sha256()
    if isinstance(filepath_or_bytes, bytes):
        hasher.update(filepath_or_bytes)
    else:
        path = Path(filepath_or_bytes)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def resolve_ffmpeg_binary() -> Optional[str]:
    """Locates the ffmpeg executable across system PATH, env vars, or imageio-ffmpeg."""
    env_path = os.getenv("FFMPEG_PATH")
    if env_path and shutil.which(env_path):
        return env_path
    
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    
    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception as e:
            logger.debug(f"imageio_ffmpeg lookup failed: {e}")
            
    return None


def detect_media_type(filepath: Path) -> str:
    """Detects media type (video, image, audio) based on extension and mime type."""
    ext = filepath.suffix.lower()
    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
    audio_exts = {".mp3", ".wav", ".aac", ".ogg", ".flac", ".m4a", ".wma"}

    if ext in video_exts:
        return "video"
    if ext in image_exts:
        return "image"
    if ext in audio_exts:
        return "audio"

    mime, _ = mimetypes.guess_type(str(filepath))
    if mime:
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"

    return "unknown"


class IngestPipeline:
    """
    Ingests URLs or uploaded files, hashes them, and splits them into
    1 fps frames and 16kHz mono audio.
    """

    def __init__(self, output_root: Union[str, Path] = "./truthlens_workspace"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_bin = resolve_ffmpeg_binary()
        if not self.ffmpeg_bin:
            logger.warning("FFmpeg binary not detected on PATH or via imageio-ffmpeg. Falling back to OpenCV where possible.")

    def download_url(self, url: str, target_dir: Path) -> Tuple[Path, Dict]:
        """
        Downloads social media video or media from Instagram, YouTube, etc. using yt-dlp.
        Returns the local file path and extracted metadata.
        """
        if yt_dlp is None:
            raise RuntimeError("yt-dlp package is required for URL downloads. Install via pip install yt-dlp.")

        target_dir.mkdir(parents=True, exist_ok=True)
        outtmpl = str(target_dir / "%(id)s.%(ext)s")

        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            downloaded_path = Path(filename)
            
            # yt-dlp sometimes merges into .mp4 or changes extension
            if not downloaded_path.exists():
                stem = downloaded_path.stem
                matches = list(target_dir.glob(f"{stem}.*"))
                if matches:
                    downloaded_path = matches[0]

            metadata = {
                "source_url": url,
                "title": info.get("title"),
                "uploader": info.get("uploader") or info.get("channel"),
                "upload_date": info.get("upload_date"),
                "duration": info.get("duration"),
                "description": info.get("description"),
                "view_count": info.get("view_count"),
                "extractor": info.get("extractor"),
            }
            return downloaded_path, metadata

    def extract_frames_opencv(self, video_path: Path, frames_dir: Path) -> Tuple[List[str], List[float], float]:
        """Fallback 1 fps frame extraction using OpenCV when FFmpeg CLI is absent."""
        if cv2 is None:
            raise RuntimeError("Neither FFmpeg nor OpenCV is available for frame extraction.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = total_frames / native_fps if native_fps > 0 else 0.0

        frame_paths: List[str] = []
        frame_timestamps: List[float] = []

        step = max(1, int(round(native_fps)))
        curr_frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame_idx % step == 0:
                frame_filename = f"frame_{extracted_count + 1:04d}.jpg"
                out_path = frames_dir / frame_filename
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                frame_paths.append(str(out_path))
                timestamp = round(curr_frame_idx / native_fps, 2)
                frame_timestamps.append(timestamp)
                extracted_count += 1
            curr_frame_idx += 1

        cap.release()
        return frame_paths, frame_timestamps, duration_seconds

    def split_video(self, video_path: Path, work_dir: Path) -> Tuple[List[str], List[float], Optional[str], float]:
        """
        Splits video into:
        1) 1 fps frames (saved as frame_0001.jpg, etc.)
        2) 16kHz mono WAV audio file (audio_16k_mono.wav)
        """
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_out_path = work_dir / "audio_16k_mono.wav"

        frame_paths: List[str] = []
        frame_timestamps: List[float] = []
        duration_seconds = 0.0
        audio_extracted_path: Optional[str] = None

        if self.ffmpeg_bin:
            # 1) Extract 1 fps frames via ffmpeg
            frame_pattern = str(frames_dir / "frame_%04d.jpg")
            cmd_frames = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(video_path),
                "-vf", "fps=1",
                "-q:v", "2",
                frame_pattern
            ]
            try:
                subprocess.run(cmd_frames, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg frame extraction failed: {e.stderr.decode('utf-8', errors='ignore')}")
                # Fallback to OpenCV
                return self._fallback_video_split(video_path, work_dir)

            extracted_files = sorted(list(frames_dir.glob("frame_*.jpg")))
            for idx, p in enumerate(extracted_files):
                frame_paths.append(str(p))
                frame_timestamps.append(float(idx))
            duration_seconds = float(len(frame_paths))

            # 2) Extract 16kHz mono audio via ffmpeg
            cmd_audio = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_out_path)
            ]
            res_audio = subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_audio.returncode == 0 and audio_out_path.exists() and audio_out_path.stat().st_size > 1000:
                audio_extracted_path = str(audio_out_path)
            else:
                logger.info("Video contains no audio track or audio extraction failed.")
                audio_extracted_path = None

        else:
            # Fallback to OpenCV for frames
            frame_paths, frame_timestamps, duration_seconds = self.extract_frames_opencv(video_path, frames_dir)
            audio_extracted_path = None

        return frame_paths, frame_timestamps, audio_extracted_path, duration_seconds

    def _fallback_video_split(self, video_path: Path, work_dir: Path) -> Tuple[List[str], List[float], Optional[str], float]:
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths, frame_timestamps, duration = self.extract_frames_opencv(video_path, frames_dir)
        return frame_paths, frame_timestamps, None, duration

    def convert_audio(self, audio_path: Path, work_dir: Path) -> Tuple[Optional[str], float]:
        """Converts arbitrary audio format to 16kHz mono PCM WAV."""
        target_audio = work_dir / "audio_16k_mono.wav"
        if self.ffmpeg_bin:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(audio_path),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(target_audio)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and target_audio.exists():
                # Estimate duration
                duration = 0.0
                try:
                    import soundfile as sf
                    info = sf.info(str(target_audio))
                    duration = info.duration
                except Exception:
                    pass
                return str(target_audio), duration
        
        # Fallback via librosa & soundfile if available
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            sf.write(str(target_audio), y, 16000, subtype='PCM_16')
            return str(target_audio), float(len(y) / 16000)
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None, 0.0

    def process(
        self,
        source: Union[str, Path],
        is_url: bool = False,
        original_filename: Optional[str] = None
    ) -> IngestResult:
        """
        Main entry point for ingestion. Accepts a social media URL or a local file path.
        Computes SHA-256 hash, organizes workspace, and runs FFmpeg splitting.
        """
        temp_dir = self.output_root / "temp_incoming"
        temp_dir.mkdir(parents=True, exist_ok=True)

        url_metadata: Dict = {}
        if is_url:
            raw_path, url_metadata = self.download_url(str(source), temp_dir)
            orig_name = raw_path.name
        else:
            raw_path = Path(source)
            if not raw_path.exists():
                raise FileNotFoundError(f"Input file not found: {raw_path}")
            orig_name = original_filename or raw_path.name

        # 1. Compute SHA-256 hash
        sha256 = compute_sha256(raw_path)
        logger.info(f"Ingested media SHA-256: {sha256}")

        # 2. Setup job work directory keyed by SHA-256 hash
        job_dir = self.output_root / sha256
        job_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        final_orig_path = job_dir / f"original_{orig_name}"
        if not final_orig_path.exists():
            shutil.copy2(str(raw_path), str(final_orig_path))

        media_type = detect_media_type(final_orig_path)
        frame_paths: List[str] = []
        frame_timestamps: List[float] = []
        audio_path: Optional[str] = None
        duration = 0.0

        if media_type == "video":
            frame_paths, frame_timestamps, audio_path, duration = self.split_video(final_orig_path, job_dir)
        elif media_type == "image":
            # Copy single image to frames directory as frame_0001.jpg
            frame_out = frames_dir / "frame_0001.jpg"
            if cv2 is not None:
                img = cv2.imread(str(final_orig_path))
                if img is not None:
                    cv2.imwrite(str(frame_out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                else:
                    shutil.copy2(str(final_orig_path), str(frame_out))
            else:
                shutil.copy2(str(final_orig_path), str(frame_out))
            frame_paths = [str(frame_out)]
            frame_timestamps = [0.0]
            duration = 0.0
            audio_path = None
        elif media_type == "audio":
            audio_path, duration = self.convert_audio(final_orig_path, job_dir)
            frame_paths = []
            frame_timestamps = []
        else:
            # Attempt video split first, if fails treat as image or audio
            try:
                frame_paths, frame_timestamps, audio_path, duration = self.split_video(final_orig_path, job_dir)
                if frame_paths:
                    media_type = "video"
            except Exception:
                media_type = "unknown"

        return IngestResult(
            sha256_hash=sha256,
            media_type=media_type,
            original_filename=orig_name,
            original_filepath=str(final_orig_path),
            work_dir=str(job_dir),
            frames_dir=str(frames_dir),
            frame_paths=frame_paths,
            frame_timestamps=frame_timestamps,
            audio_path=audio_path,
            duration_seconds=duration,
            metadata=url_metadata
        )
