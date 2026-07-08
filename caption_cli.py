import os
import sys

# Try to dynamically add nvidia CUDA paths to DLL search directories on Windows
if sys.platform == "win32":
    import site
    paths = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        paths.append(site.getusersitepackages())
    for path in paths:
        nvidia_path = os.path.join(path, "nvidia")
        if os.path.isdir(nvidia_path):
            for root, dirs, files in os.walk(nvidia_path):
                if any(f.endswith(".dll") for f in files):
                    try:
                        os.add_dll_directory(root)
                        os.environ["PATH"] = root + os.path.pathsep + os.environ["PATH"]
                    except Exception:
                        pass

import queue
import time
import argparse
import threading
import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import pyttsx3
import ctranslate2

def is_cuda_available():
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

# Add local bin to PATH for FFmpeg accessibility
project_root = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(project_root, "bin")
if os.path.isdir(bin_path):
    os.environ["PATH"] = bin_path + os.path.pathsep + os.environ["PATH"]

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Shared Queues
audio_queue = queue.Queue()
transcription_queue = queue.Queue()
running = False

def list_devices():
    """Query and print all active WASAPI capture/loopback devices."""
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_index = wasapi_info["index"]
    except Exception:
        print("Error: Windows WASAPI Host API is not available.")
        p.terminate()
        sys.exit(1)
        
    devices = []
    print("\nAvailable WASAPI Audio Sources:")
    print("-" * 60)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["hostApi"] != wasapi_index or info["maxInputChannels"] <= 0:
            continue
            
        label = "System Loopback (Speakers)" if info.get("isLoopbackDevice", False) else "Microphone"
        print(f"Index {info['index']}: {info['name']} ({label})")
        devices.append(info)
        
    print("-" * 60)
    p.terminate()
    return devices

def resample_chunk(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    orig_indices = np.arange(len(audio))
    target_indices = np.linspace(0, len(audio) - 1, int(duration * target_sr))
    return np.interp(target_indices, orig_indices, audio).astype(np.float32)

def audio_pipeline_loop(native_rate, native_channels, sensitivity, timeout):
    """Downmix, resample, and run VAD on incoming audio bytes."""
    active_speech_buffer = []
    silence_samples_count = 0
    
    while running:
        try:
            raw_bytes = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        chunk = np.frombuffer(raw_bytes, dtype=np.float32)
        if len(chunk) == 0:
            continue
            
        # Convert stereo to mono
        if native_channels > 1:
            chunk = chunk.reshape(-1, native_channels).mean(axis=1)
            
        # Resample to 16000Hz
        resampled = resample_chunk(chunk, native_rate, 16000)
        
        # VAD Energy check
        rms = np.sqrt(np.mean(resampled ** 2))
        
        if rms > sensitivity:
            active_speech_buffer.append(resampled)
            silence_samples_count = 0
        else:
            if active_speech_buffer:
                active_speech_buffer.append(resampled)
                silence_samples_count += len(resampled)
                
                silence_seconds = silence_samples_count / 16000
                if silence_seconds >= timeout:
                    # Finalize segment
                    audio_data = np.concatenate(active_speech_buffer)
                    transcription_queue.put(audio_data)
                    active_speech_buffer = []
                    silence_samples_count = 0
                    
        # Max segment safeguard (15 seconds)
        if active_speech_buffer:
            buffer_duration = sum(len(x) for x in active_speech_buffer) / 16000
            if buffer_duration > 15.0:
                audio_data = np.concatenate(active_speech_buffer)
                transcription_queue.put(audio_data)
                
                overlap_samples = int(0.5 * 16000)
                if len(audio_data) > overlap_samples:
                    active_speech_buffer = [audio_data[-overlap_samples:]]
                else:
                    active_speech_buffer = []
                silence_samples_count = 0

def run_tts(text):
    """Narrate the finalized phrase using Windows SAPI5."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
        del engine
    except Exception as e:
        print(f"\n[TTS Error]: {e}", file=sys.stderr)

def transcription_loop(model, tts_enabled):
    """Run inference on completed audio blocks and output to console/TTS."""
    while running:
        try:
            audio_data = transcription_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        try:
            segments, _ = model.transcribe(audio_data, beam_size=2, vad_filter=True)
            text = " ".join([segment.text for segment in segments]).strip()
            
            # Skip empty or default noise segments
            if not text or text.lower() in [".", "you", "thank you", "thank you.", "bye", "bye."]:
                continue
                
            # Print text to stdout immediately (which screen readers read as a terminal update)
            print(f"\nCaption: {text}")
            sys.stdout.flush()
            
            if tts_enabled:
                # Narrate in background
                threading.Thread(target=run_tts, args=(text,), daemon=True).start()
                
        except Exception as e:
            print(f"\n[ASR Error]: {e}", file=sys.stderr)

def main():
    global running
    
    parser = argparse.ArgumentParser(description="Accessible Live Captioning CLI Companion")
    parser.add_argument("--device", type=int, default=-1, help="WASAPI device index to record from")
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small", "medium"], help="Whisper model size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference instead of CUDA")
    parser.add_argument("--tts", action="store_true", help="Enable automatic Text-To-Speech read-out")
    parser.add_argument("--sensitivity", type=float, default=0.006, help="RMS energy sensitivity threshold (default: 0.006)")
    parser.add_argument("--timeout", type=float, default=1.5, help="Seconds of silence to trigger phrase finalization (default: 1.5)")
    args = parser.parse_args()
    
    print("==================================================")
    print("  Accessible Live Captioning (ALC) CLI Companion  ")
    print("==================================================")
    
    # Identify device
    devices = list_devices()
    device_idx = args.device
    if device_idx == -1:
        while True:
            try:
                input_str = input("Select an Audio Source Index to start: ").strip()
                device_idx = int(input_str)
                if any(d['index'] == device_idx for d in devices):
                    break
                print("Invalid index. Please select one from the list above.")
            except ValueError:
                print("Please enter a valid integer index.")
                
    # Detect appropriate device details
    p = pyaudio.PyAudio()
    selected_device = p.get_device_info_by_index(device_idx)
    p.terminate()
    
    device_label = "System Loopback" if selected_device.get("isLoopbackDevice", False) else "Microphone"
    print(f"\nSelected Device: {selected_device['name']} ({device_label})")
    
    # Load Whisper Model
    device_type = "cpu" if args.cpu or not is_cuda_available() else "cuda"
    compute_type = "int8" if device_type == "cpu" else "float16"
    print(f"Loading Whisper model '{args.model}' on {device_type.upper()} ({compute_type})...")
    
    try:
        model = WhisperModel(args.model, device=device_type, compute_type=compute_type)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Start Pipeline
    running = True
    
    # Start thread targets
    native_rate = int(selected_device["defaultSampleRate"])
    native_channels = selected_device["maxInputChannels"]
    
    pipeline_thread = threading.Thread(
        target=audio_pipeline_loop, 
        args=(native_rate, native_channels, args.sensitivity, args.timeout),
        daemon=True
    )
    trans_thread = threading.Thread(
        target=transcription_loop, 
        args=(model, args.tts),
        daemon=True
    )
    
    pipeline_thread.start()
    trans_thread.start()
    
    # Setup PyAudio Recording Stream
    pa = pyaudio.PyAudio()
    
    def callback(in_data, frame_count, time_info, status):
        if running:
            audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=native_channels,
            rate=native_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=1024,
            stream_callback=callback
        )
        
        print("\n>>> LIVE TRANSCRIPTION ACTIVE. PRESS CTRL+C TO STOP. <<<\n")
        
        # Keep main thread alive
        while True:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        running = False
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
        print("Finished.")

if __name__ == "__main__":
    main()
