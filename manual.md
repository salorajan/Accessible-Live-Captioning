# Accessible Live Captioning (ALC) - User Manual

Welcome to the **Accessible Live Captioning (ALC)** system. This application is designed to capture audio from either a microphone or your system speakers (loopback), generate real-time captions using a local offline Whisper AI model, display them in an accessible interface, and optionally read them aloud using local Text-To-Speech (TTS).

---

## 1. Quick Start

The project contains two versions of the captioning application:

### Option A: Desktop GUI (Recommended for Screen Readers)
1. Double-click **`run_app.bat`** in the project directory.
2. A native Windows frame will open.
3. Use the **Audio Source** menu to select your microphone or system speakers.
4. Press **F5** (or click **Start Captioning**) to begin.
5. Press **F6** (or click **Stop Captioning**) to stop.
*   **Accessibility Features:** Built using `wxPython`, which maps directly to Windows native accessibility APIs (MSAA/UIA). The **Transcription History** is a native list box that screen readers (NVDA, JAWS, Narrator) can read and navigate item-by-item using the Up and Down arrow keys.

### Option B: Command-Line Interface (CLI Companion)
1. Double-click **`run_cli.bat`** in the project directory.
2. A console window will print a list of available audio devices.
3. Enter the index number of the source you wish to capture (e.g., `10` for Speakers [Loopback]).
4. The offline Whisper model will load, and live captions will print to the console.
5. Press **Ctrl + C** to stop.

---

## 2. Preventing the Audio Feedback Loop (Critical)

If you are using a screen reader (NVDA, JAWS, Narrator) and choose **System Loopback (Speakers)** as your source, the application will record the screen reader's own voice as it announces new captions, creating an infinite loop.

To prevent this:

1.  **For Microphone Input:**
    *   Set the source to your **Microphone Array**.
    *   **Always wear headphones** so the screen reader's speech does not leak from your speakers back into the microphone.
2.  **For System Audio (Zoom, YouTube, Meetings):**
    *   Set the source to **Speakers [Loopback]**.
    *   Temporarily mute your screen reader's voice while transcribing (e.g., in NVDA, press `NVDA + S` to switch to **"Speech Mode: Off"** or **"Speech Mode: Beeps"**).
    *   Alternatively, route your screen reader's voice output to a different audio device (like a USB headset) and keep the system audio on speakers, capturing only the speaker stream.

---

## 3. Configuration & Options

Both applications support the following settings:

*   **Whisper Model Size:**
    *   `tiny` (Fastest, low accuracy, small download)
    *   `base` (Default - balanced speed and accuracy, highly recommended)
    *   `small` (More accurate, requires more memory)
    *   `medium` (Highly accurate, slow execution, large file size)
*   **Device Type:** Choose **GPU (CUDA)** to use your dedicated NVIDIA RTX 500 GPU for near-instant transcription, or **CPU** (runs slower but has lower resource usage).
*   **Voice Sensitivity:** Adjusts the energy threshold to detect speech. High sensitivity detects quiet speech; Low sensitivity ignores background hums/noise.
*   **Pause Duration (Silence Timeout):** The duration of silence (e.g., 1.5 seconds) required to determine that a phrase is finished, finalizing it and adding it to the history.
*   **Text Size (GUI only):** Set font sizes from Small (12pt) to Huge (32pt) to assist users with low vision.
*   **Read Captions Automatically (TTS):** When checked, the application reads finalized captions aloud using the Windows native offline SAPI5 speech engine.

---

## 4. Technical Information (For Developers)

The project dependencies are fully configured in the virtual environment located at `C:\salo\jeff\env_caption\`.

*   **GUI Framework:** `wxPython` (wraps Win32 native controls).
*   **ASR Engine:** `faster-whisper` (runs CTranslate2 backend on CPU or GPU with CUDA).
*   **Audio Capture:** `pyaudiowpatch` (pyaudio fork exposing WASAPI loopback streams).
*   **TTS Engine:** `pyttsx3` (initialized per-phrase to prevent driver hangs).
*   **NVIDIA CUDA Library Packages:** `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, and `nvidia-cuda-nvrtc-cu12` are installed to run Whisper on the GPU.

### Adding New Features
*   To modify the GUI version, edit [caption_app.py](file:///C:/salo/jeff/caption/caption_app.py).
*   To modify the CLI version, edit [caption_cli.py](file:///C:/salo/jeff/caption/caption_cli.py).
*   All dynamic libraries (FFmpeg binaries) are stored in the `bin/` directory and loaded into path automatically by the scripts.
