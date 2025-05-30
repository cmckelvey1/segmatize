import argparse
import csv
import os
import subprocess
import json
from datetime import datetime
import multiprocessing

from pydub import AudioSegment
from video_converter import convert_video_to_audio, convert_m4a_to_mp3

# ------------------------------
# Helper Functions
# ------------------------------

def create_date_subdir(base_out):
    """
    Create a subdirectory structure based on current date:
    <base_out>/<year>/<month-zero-padded>/<day-zero-padded>
    """
    now = datetime.now()
    sub_dir = os.path.join(base_out, str(now.year), f"{now.month:02d}", f"{now.day:02d}")
    os.makedirs(sub_dir, exist_ok=True)
    return sub_dir

def split_audio(audio_path, num_segments, output_dir):
    """
    Splits the audio file into N equally sized segments.
    Returns a list of the output audio segment paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_mp3(audio_path)
    total_duration_ms = len(audio)
    segment_duration_ms = total_duration_ms // num_segments
    segments = []
    for i in range(num_segments):
        start_ms = i * segment_duration_ms
        end_ms = (i + 1) * segment_duration_ms if i < num_segments - 1 else total_duration_ms
        segment = audio[start_ms:end_ms]
        output_path = os.path.join(output_dir, f"segment_{i+1}.mp3")
        segment.export(output_path, format="mp3")
        segments.append(output_path)
    return segments

def run_whisper_on_segment(audio_segment_path, output_dir, gpu_id):
    """
    Runs Whisper on a single audio segment, specifying the GPU to use.
    Returns the path to the JSON output.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_segment_path))[0]
    json_file = os.path.join(output_dir, base + ".json")
    cmd = [
        "whisper", audio_segment_path,
        "--model", "large-v3",
        "--output_format", "json",
        "--output_dir", output_dir,
        "--device", f"cuda:{gpu_id}"  # Specify the GPU device
    ]
    print(f"[Whisper] Running on '{audio_segment_path}' using GPU {gpu_id}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error running Whisper on GPU {gpu_id}]: {e}")
        # Handle the error appropriately, perhaps by retrying on a different GPU or CPU
    return json_file

def parallel_whisper(audio_segments, output_base_dir, num_gpus):
    """
    Runs Whisper in parallel on the provided audio segments, assigning each to a GPU.
    Returns a list of paths to the resulting JSON transcription files.
    """
    pool = multiprocessing.Pool(processes=num_gpus)
    results = []
    for i, segment_path in enumerate(audio_segments):
        segment_output_dir = os.path.join(output_base_dir, f"segment_{i+1}")
        gpu_id = i % num_gpus  # Cycle through available GPUs
        results.append(pool.apply_async(run_whisper_on_segment, args=(segment_path, segment_output_dir, gpu_id)))
    pool.close()
    pool.join()
    return [res.get() for res in results]

def combine_segment_transcriptions(json_files):
    """
    Reads and combines the Whisper JSON transcriptions from multiple segments,
    sorting them by their original order.
    Returns the final transcript as a single string.
    """
    segment_data = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            segments = data.get("segments", [])
            segment_number = int(os.path.basename(json_file).split("_")[1].split(".")[0])
            segment_data.append({"segment": segment_number, "segments": segments})

    # Sort segment data by segment number
    segment_data.sort(key=lambda x: x["segment"])

    final_text = ""
    for data in segment_data:
        # Sort segments within each file by start time
        data["segments"].sort(key=lambda x: x.get("start", 0))
        final_text += "\n".join(seg.get("text", "").strip() for seg in data["segments"] if seg.get("text", "").strip()) + "\n"

    return final_text.strip()

def get_num_gpus():
    """
    Attempts to determine the number of available GPUs.
    Returns 1 if detection fails or no GPUs are found.
    """
    try:
        result = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
        output = result.stdout.strip()
        if output:
            return len(output.split("\n"))
        else:
            return 1
    except FileNotFoundError:
        print("nvidia-smi not found. Assuming 1 GPU.")
        return 1
    except Exception as e:
        print(f"Error detecting GPUs: {e}. Assuming 1 GPU.")
        return 1

def run_whisper_on_audio_parallel(audio_file, transcription_output_base_dir):
    """
    Splits the audio, runs Whisper in parallel, and combines the transcripts,
    explicitly assigning each process to a GPU.
    """
    num_gpus = get_num_gpus()
    print(f"Detected {num_gpus} GPUs. Splitting audio into {num_gpus} segments.")

    audio_split_dir = os.path.join(transcription_output_base_dir, "audio_segments")
    audio_segments = split_audio(audio_file, num_gpus, audio_split_dir)

    parallel_output_dir = os.path.join(transcription_output_base_dir, "parallel_transcriptions")
    json_files = parallel_whisper(audio_segments, parallel_output_dir, num_gpus)

    final_transcript = combine_segment_transcriptions(json_files)
    return final_transcript

# ------------------------------
# Main Processing Function
# ------------------------------

def process_line(row, base_out):
    """
    Process one CSV row.
    Expected CSV row fields (comma-separated):
      1. Video/Audio file location (MP4 or M4A)
      2. Names list (separated by '_') – for reference
      3. Folder name (used to name the conversation subdirectory)
      4. Description file location (not used in processing)
    """
    if len(row) < 4:
        print("Skipping row (not enough entries):", row)
        return

    input_file_path = row[0].strip()
    # Capture names_list if needed (currently not used)
    names_list = row[1].strip().split("_")
    folder_name = row[2].strip()
    description_file = row[3].strip()  # Not used in this script

    # Create dated subdirectory: <base_out>/<year>/<month>/<day>
    date_subdir = create_date_subdir(base_out)
    conversation_dir = os.path.join(date_subdir, folder_name)
    os.makedirs(conversation_dir, exist_ok=True)

    # Create subdirectories for audio and transcription
    audio_dir = os.path.join(conversation_dir, "audio")
    transcription_dir = os.path.join(conversation_dir, "transcription", "speaker_0")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcription_dir, exist_ok=True)

    # Step 1: Convert input file to audio (MP3)
    full_audio_path = os.path.join(audio_dir, "full_audio.mp3")
    
    # Check file extension and convert accordingly
    file_extension = os.path.splitext(input_file_path)[1].lower()
    
    if file_extension == ".mp4":
        print(f"Processing MP4 video file: {input_file_path}")
        audio_file_path = convert_video_to_audio(input_file_path, full_audio_path)
    elif file_extension == ".m4a":
        print(f"Processing M4A audio file: {input_file_path}")
        audio_file_path = convert_m4a_to_mp3(input_file_path, full_audio_path)
    else:
        print(f"Unsupported file type: {file_extension}. Skipping file: {input_file_path}")
        return

    if not audio_file_path:
        print(f"Audio conversion failed for {input_file_path}. Skipping.")
        return

    # Step 2: Run Whisper on the full audio file in parallel.
    final_transcript = run_whisper_on_audio_parallel(audio_file_path, transcription_dir)

    # Output the final transcript to stdout.
    print("\n----- Final Conversation Transcript -----\n")
    print(final_transcript)
    print("\n-----------------------------------------\n")

    transcript_output_file = os.path.join(transcription_dir, "full_transcript.txt")
    with open(transcript_output_file, "w+") as f:
        f.write(final_transcript)


# ------------------------------
# Main Script
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process a CSV file with video info: convert video to audio, run Whisper for transcription in parallel, and output the final transcript."
    )
    parser.add_argument("-f", "--file", required=True,
                        help="Input CSV file (each line: video_file,names_list,folder_name,description_file)")
    parser.add_argument("-o", "--output", required=True,
                        help="Base output directory")

    args = parser.parse_args()

    with open(args.file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            process_line(row, args.output)

if __name__ == "__main__":
    main()