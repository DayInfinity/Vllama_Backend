from pathlib import Path
import subprocess
import imageio_ffmpeg


def ensure_mp4(video_path: str) -> str:
    """
    Ensures video is in MP4 format, converting from MOV if necessary.
    
    Args:
        video_path: Path to the video file (.mp4 or .mov)
    
    Returns:
        str: Path to MP4 video file
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is not .mp4 or .mov
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Already MP4 - return as is
    if video_path.suffix.lower() == ".mp4":
        print(f"[INFO] Already MP4: {video_path.name}")
        return str(video_path)

    # Only convert MOV files
    if video_path.suffix.lower() != ".mov":
        raise ValueError(
            f"Unsupported video format: {video_path.suffix}. "
            "Only .mov and .mp4 files are supported."
        )

    # Create MP4 output path
    mp4_path = video_path.with_suffix(".mp4")

    # Get FFmpeg executable
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"[INFO] Converting {video_path.name} → {mp4_path.name}")

    # Build FFmpeg command
    cmd = [
        ffmpeg_exe,
        "-y",                       # Overwrite output file if exists
        "-loglevel", "error",       # Only show errors
        "-i", str(video_path),      # Input file
        "-c:v", "libx264",          # Video codec
        "-c:a", "aac",              # Audio codec
        "-movflags", "+faststart",  # Enable fast start for web playback
        str(mp4_path)               # Output file
    ]

    try:
        # Run FFmpeg conversion
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        print(f"[SUCCESS] Converted to MP4: {mp4_path}")
        
        # Verify the output file was created
        if not mp4_path.exists():
            raise RuntimeError(f"Conversion completed but output file not found: {mp4_path}")
        
        return str(mp4_path)
    
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")