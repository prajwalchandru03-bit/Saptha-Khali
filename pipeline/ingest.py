"""
TruthLens Ingestion Module (Instagram & Social Media Optimized)
Handles:
- Social media URL downloads (Instagram Reels, YouTube Shorts, etc.) via yt-dlp
- Realistic user-agent headers, fallback MP4 streams, and clean caption/uploader extraction
- User-friendly error handling for Instagram login walls and rate limits
- Cryptographic SHA-256 provenance hashing
- Fast capped keyframe extraction (MAX_VIDEO_FRAMES_ANALYSIS default 10) for CPU efficiency
- 16kHz mono WAV audio extraction via FFmpeg/soundfile
"""

import hashlib
import logging
import mimetypes
import os
import re
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

DEFAULT_MAX_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES_ANALYSIS", "10"))


class IngestionError(Exception):
    """Raised when media ingestion fails (e.g. Instagram login wall, private post, rate limit)."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


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
    Ingests Instagram Reels, YouTube Shorts, URLs or uploaded files,
    hashes them, and extracts capped keyframes (default 10) and 16kHz mono audio.
    """

    def __init__(
        self,
        output_root: Union[str, Path] = "./truthlens_workspace",
        max_keyframes: int = DEFAULT_MAX_FRAMES
    ):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_keyframes = max_keyframes
        self.ffmpeg_bin = resolve_ffmpeg_binary()
        if not self.ffmpeg_bin:
            logger.warning("FFmpeg binary not detected on PATH or via imageio-ffmpeg. Falling back to OpenCV where possible.")

    def download_url(self, url: str, target_dir: Path) -> Tuple[Path, Dict]:
        """
        Downloads social media video from Instagram Reels, YouTube, etc. using yt-dlp.
        Supports shortcode extraction for Instagram (trying /p/{shortcode}/ first, then /reel/{shortcode}/).
        Configured with verified options: force_ipv4, retries, cookiefile, and carousel entry handling.
        """
        if yt_dlp is None:
            raise RuntimeError("yt-dlp package is required for URL downloads. Install via pip install yt-dlp.")

        target_dir.mkdir(parents=True, exist_ok=True)
        outtmpl = str(target_dir / "%(id)s.%(ext)s")

        # Load cookies.txt if present in the workspace root
        cookie_path = Path(__file__).resolve().parent.parent / "cookies.txt"

        # Verified yt-dlp options (no custom user_agent, http_headers, or impersonate)
        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "force_ipv4": True,
            "playlist_items": "1",
            "retries": 3,
            "extractor_retries": 3,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        if cookie_path.exists():
            ydl_opts["cookiefile"] = str(cookie_path)
            logger.info(f"Loaded Instagram/YouTube cookies from: {cookie_path}")

        # Resolve candidate URLs for Instagram posts/reels
        candidate_urls: List[str] = []
        if "instagram.com" in url.lower():
            # Strip query parameters (e.g. ?igsh=...)
            clean_url = url.split("?")[0].rstrip("/")
            match = re.search(r"/(?:reel|reels|p)/([A-Za-z0-9_-]+)", clean_url)
            if match:
                shortcode = match.group(1)
                # Always try /p/{shortcode}/ first, then fall back to /reel/{shortcode}/
                candidate_urls = [
                    f"https://www.instagram.com/p/{shortcode}/",
                    f"https://www.instagram.com/reel/{shortcode}/",
                ]
            else:
                candidate_urls = [clean_url]
        else:
            candidate_urls = [url]

        info = None
        last_error = None
        attempt_url = candidate_urls[0]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                for idx, cand_url in enumerate(candidate_urls):
                    attempt_url = cand_url
                    try:
                        logger.info(f"Extracting video info with yt-dlp: {cand_url}")
                        extracted = ydl.extract_info(cand_url, download=True)
                        if extracted:
                            info = extracted
                            break
                    except yt_dlp.utils.DownloadError as de:
                        last_error = de
                        if idx < len(candidate_urls) - 1:
                            logger.warning(
                                f"yt-dlp attempt failed for '{cand_url}': {de}. "
                                f"Falling back to candidate: '{candidate_urls[idx + 1]}'."
                            )
                            continue
                        else:
                            # Log exact raw yt-dlp error via logger.error
                            logger.error(f"yt-dlp extraction failed for '{cand_url}': {de}")
                            raise
                    except Exception as e:
                        last_error = e
                        if idx < len(candidate_urls) - 1:
                            logger.warning(
                                f"yt-dlp error for '{cand_url}': {e}. "
                                f"Falling back to candidate: '{candidate_urls[idx + 1]}'."
                            )
                            continue
                        else:
                            logger.error(f"yt-dlp extraction failed for '{cand_url}': {e}", exc_info=True)
                            raise

                if not info:
                    raise IngestionError(
                        "Unable to extract video information from the provided URL. "
                        "The post may be private or removed. Please upload the MP4 file directly.",
                        status_code=400
                    )

                # Instagram carousel posts or playlists return a list of entries
                target_info = info
                if "entries" in info and info["entries"]:
                    entry = info["entries"][0]
                    if entry:
                        target_info = entry

                filename = ydl.prepare_filename(target_info)
                downloaded_path = Path(filename)

                # Match file if yt-dlp merged into .mp4
                if not downloaded_path.exists():
                    stem = downloaded_path.stem
                    matches = list(target_dir.glob(f"{stem}.*"))
                    if matches:
                        downloaded_path = matches[0]

                if not downloaded_path.exists():
                    # Fallback check for newly written video files in target_dir
                    candidates = sorted(
                        list(target_dir.glob("*.mp4")) + list(target_dir.glob("*.mkv")) + list(target_dir.glob("*.webm")),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )
                    if candidates:
                        downloaded_path = candidates[0]

                if not downloaded_path.exists():
                    raise IngestionError(
                        "Downloaded file not found on disk. Please upload the MP4 file directly.",
                        status_code=400
                    )

                # Clean extraction of caption and creator info
                description = (
                    target_info.get("description")
                    or target_info.get("title")
                    or info.get("description")
                    or info.get("title")
                    or ""
                )
                uploader = (
                    target_info.get("uploader")
                    or target_info.get("channel")
                    or target_info.get("creator")
                    or target_info.get("uploader_id")
                    or info.get("uploader")
                    or info.get("channel")
                    or info.get("creator")
                    or info.get("uploader_id")
                    or "Unknown Uploader"
                )

                metadata = {
                    "source_url": url,
                    "target_url": attempt_url,
                    "title": target_info.get("title") or info.get("title") or "",
                    "description": str(description).strip(),
                    "uploader": str(uploader).strip(),
                    "upload_date": target_info.get("upload_date") or info.get("upload_date"),
                    "duration": target_info.get("duration") or info.get("duration"),
                    "view_count": target_info.get("view_count") or info.get("view_count"),
                    "extractor": target_info.get("extractor") or info.get("extractor"),
                }
                return downloaded_path, metadata

        except yt_dlp.utils.DownloadError as de:
            err_msg = str(de).lower()
            if any(term in err_msg for term in ["login", "rate-limit", "rate limit", "checkpoint", "blocked", "restricted", "private", "sign in", "429"]):
                raise IngestionError(
                    "Instagram requires login or has rate-limited this URL. "
                    "Please download the Reel/video and upload the MP4 file directly via the upload tab.",
                    status_code=400
                )
            raise IngestionError(
                f"Failed to access video at {url}. "
                "Instagram or the platform may restrict public access to this Reel. "
                "Please download the Reel and upload the MP4 file directly.",
                status_code=400
            )

        except IngestionError:
            raise

        except Exception as e:
            err_msg = str(e).lower()
            if "instagram" in err_msg or "login" in err_msg or "rate" in err_msg:
                raise IngestionError(
                    "Instagram requires login or has rate-limited this URL. "
                    "Please download the Reel/video and upload the MP4 file directly via the upload tab.",
                    status_code=400
                )
            raise IngestionError(
                f"Could not retrieve video from link: {str(e)}. "
                "Please upload the video file directly.",
                status_code=400
            )

    def extract_keyframes_opencv(
        self,
        video_path: Path,
        frames_dir: Path,
        max_keyframes: Optional[int] = None
    ) -> Tuple[List[str], List[float], float]:
        """
        Fast capped keyframe extraction using OpenCV:
        Evenly spaces up to max_keyframes across the duration so Reels process fast on CPU.
        """
        if cv2 is None:
            raise RuntimeError("Neither FFmpeg nor OpenCV is available for frame extraction.")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = total_frames / native_fps if native_fps > 0 else 0.0

        target_count = max_keyframes or self.max_keyframes
        if total_frames <= 0 or duration_seconds <= 0:
            target_indices = [0]
        elif total_frames <= target_count:
            target_indices = list(range(total_frames))
        else:
            step = total_frames / float(target_count)
            target_indices = [int(i * step) for i in range(target_count)]

        frame_paths: List[str] = []
        frame_timestamps: List[float] = []

        for count, frame_idx in enumerate(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_filename = f"frame_{count + 1:04d}.jpg"
            out_path = frames_dir / frame_filename
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_paths.append(str(out_path))

            timestamp = round(frame_idx / native_fps, 2)
            frame_timestamps.append(timestamp)

        cap.release()
        return frame_paths, frame_timestamps, duration_seconds

    def split_video(
        self,
        video_path: Path,
        work_dir: Path,
        max_keyframes: Optional[int] = None
    ) -> Tuple[List[str], List[float], Optional[str], float]:
        """
        Splits video into:
        1) Evenly spaced keyframes (capped at max_keyframes, default 10)
        2) 16kHz mono WAV audio file (audio_16k_mono.wav)
        """
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_out_path = work_dir / "audio_16k_mono.wav"

        target_k = max_keyframes or self.max_keyframes

        # 1. Keyframe extraction using OpenCV (capped at target_k)
        try:
            frame_paths, frame_timestamps, duration_seconds = self.extract_keyframes_opencv(
                video_path, frames_dir, max_keyframes=target_k
            )
        except Exception as e:
            logger.warning(f"OpenCV frame extraction encountered an issue: {e}. Trying FFmpeg fallback...")
            frame_paths, frame_timestamps, duration_seconds = self._ffmpeg_capped_split(
                video_path, frames_dir, target_k
            )

        # 2. Extract 16kHz mono audio via FFmpeg
        audio_extracted_path: Optional[str] = None
        if self.ffmpeg_bin:
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
        else:
            try:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(str(video_path), sr=16000, mono=True)
                if len(y) > 0:
                    sf.write(str(audio_out_path), y, 16000, subtype='PCM_16')
                    audio_extracted_path = str(audio_out_path)
            except Exception:
                audio_extracted_path = None

        return frame_paths, frame_timestamps, audio_extracted_path, duration_seconds

    def _ffmpeg_capped_split(
        self,
        video_path: Path,
        frames_dir: Path,
        target_k: int
    ) -> Tuple[List[str], List[float], float]:
        """FFmpeg fallback for extracting target_k keyframes."""
        if not self.ffmpeg_bin:
            return [], [], 0.0

        frame_pattern = str(frames_dir / "frame_%04d.jpg")
        cmd_frames = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vf", "fps=1",
            "-q:v", "2",
            frame_pattern
        ]
        subprocess.run(cmd_frames, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        extracted = sorted(list(frames_dir.glob("frame_*.jpg")))

        if not extracted:
            return [], [], 0.0

        # Subsample to target_k
        if len(extracted) > target_k:
            step = len(extracted) / float(target_k)
            selected = [extracted[int(i * step)] for i in range(target_k)]
        else:
            selected = extracted

        frame_paths = [str(p) for p in selected]
        frame_timestamps = [float(i) for i in range(len(selected))]
        return frame_paths, frame_timestamps, float(len(extracted))

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
                duration = 0.0
                try:
                    import soundfile as sf
                    info = sf.info(str(target_audio))
                    duration = info.duration
                except Exception:
                    pass
                return str(target_audio), duration

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
        Main entry point for ingestion. Accepts an Instagram/YouTube URL or uploaded file.
        Computes SHA-256 hash, organizes workspace, and extracts capped keyframes.
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
            frame_paths, frame_timestamps, audio_path, duration = self.split_video(
                final_orig_path, job_dir, max_keyframes=self.max_keyframes
            )
        elif media_type == "image":
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
            try:
                frame_paths, frame_timestamps, audio_path, duration = self.split_video(
                    final_orig_path, job_dir, max_keyframes=self.max_keyframes
                )
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
            duration_seconds=round(duration, 2),
            metadata=url_metadata
        )
