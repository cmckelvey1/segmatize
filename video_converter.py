import os
import subprocess

def convert_video_to_audio(video_path, output_audio_path=None):
    """
    Convert the given video file to MP3 format using ffmpeg.
    
    Args:
        video_path: Path to the input video file
        output_audio_path: Optional path for the output audio file. 
                          If None, creates an MP3 file in the same directory as the video.
    
    Returns:
        str: Path to the resulting audio file
    """
    if output_audio_path is None:
        # Create output path in same directory as video with .mp3 extension
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_audio_path = os.path.join(video_dir, f"{video_name}.mp3")
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-q:a", "0", "-map", "a", output_audio_path
    ]
    print(f"[FFmpeg] Converting video '{video_path}' to audio '{output_audio_path}'")
    subprocess.run(cmd, check=True)
    
    return output_audio_path

def convert_m4a_to_mp3(m4a_path, output_mp3_path):
    """
    Convert the given M4A audio file to MP3 format using ffmpeg.

    Args:
        m4a_path: Path to the input M4A file.
        output_mp3_path: Path for the output MP3 file.

    Returns:
        str: Path to the resulting MP3 file.
    """
    cmd = [
        "ffmpeg", "-y", "-i", m4a_path,
        "-codec:a", "libmp3lame", "-q:a", "2", output_mp3_path
    ]
    print(f"[FFmpeg] Converting M4A '{m4a_path}' to MP3 '{output_mp3_path}'")
    subprocess.run(cmd, check=True)
    return output_mp3_path