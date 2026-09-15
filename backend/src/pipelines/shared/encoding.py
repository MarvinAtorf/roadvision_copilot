import subprocess
from pathlib import Path


def reencode_to_h264(video_path: Path) -> None:
    """Re-encode a video in place to H.264.

    cv2.VideoWriter with 'mp4v' produces a .mp4 that most browsers reject,
    so the annotated output is converted before it is served.
    """
    video_path = Path(video_path)
    h264_path = video_path.with_stem(video_path.stem + "_h264")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(h264_path),
        ],
        check=True,
    )

    video_path.unlink()
    h264_path.rename(video_path)
