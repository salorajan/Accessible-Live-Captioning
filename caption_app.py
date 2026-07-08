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

import threading
import queue
import time
import traceback
import numpy as np
import wx
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import pyttsx3
import ctranslate2

def is_cuda_available():
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False

# Add local bin to PATH so dependencies like FFmpeg can be located if needed
project_root = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.join(project_root, "bin")
if os.path.isdir(bin_path):
    os.environ["PATH"] = bin_path + os.path.pathsep + os.environ["PATH"]

# Set HF_HUB_DISABLE_SYMLINKS_WARNING to suppress symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Sensitivity & Timeout Mapping
SENSITIVITY_MAP = {
    "High (Quiet Speech)": 0.002,
    "Medium (Standard)": 0.006,
    "Low (Ignore Noise)": 0.015
}

TIMEOUT_MAP = {
    "Short (0.8s)": 0.8,
    "Medium (1.5s)": 1.5,
    "Long (2.5s)": 2.5
}

FONT_SIZES = {
    "Small (12pt)": 12,
    "Medium (16pt)": 16,
    "Large (20pt)": 20,
    "Extra Large (24pt)": 24,
    "Huge (32pt)": 32
}

class CaptionFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(800, 600))
        
        self.running = False
        self.recording_active = False
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Whisper model details
        self.whisper_model = None
        self.current_model_size = "base"
        self.current_device = "cuda" if is_cuda_available() else "cpu"
        self.model_loading = False
        
        # Pipeline queues
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Audio capturing configurations
        self.selected_device_index = -1
        self.sensitivity = "Medium (Standard)"
        self.silence_timeout = "Medium (1.5s)"
        self.font_size_name = "Medium (16pt)"
        self.tts_enabled = False
        
        # Active buffers for speech
        self.active_speech_buffer = []
        self.silence_samples_count = 0
        self.last_live_transcribe_time = 0
        self.live_update_interval = 1.0  # seconds
        
        # GUI construction
        self.init_ui()
        
        # Load the default Whisper model in the background
        self.trigger_model_load()
        
        # Keyboard Shortcuts (Accelerator Table)
        # F5 = Start, F6 = Stop, Ctrl+L = Clear, Ctrl+E = Export
        self.setup_shortcuts()
        
        # Bind close event
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def init_ui(self):
        panel = wx.Panel(self)
        self.panel = panel
        
        # Main layout sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 1. Transcript History ListBox (Accessible native control)
        # Accessible Description helps screen readers identify what this is
        self.listbox_label = wx.StaticText(panel, label="Transcription History (Use Arrow Keys to Navigate):")
        main_sizer.Add(self.listbox_label, 0, wx.ALL | wx.EXPAND, 8)
        
        self.history_listbox = wx.ListBox(panel, style=wx.LB_SINGLE | wx.LB_NEEDED_SB)
        self.history_listbox.SetToolTip("List of completed captions. Use arrow keys to navigate.")
        main_sizer.Add(self.history_listbox, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, 8)
        
        # 2. Live Caption Box (For real-time, in-progress updates)
        self.live_label = wx.StaticText(panel, label="Live Caption (In Progress):")
        main_sizer.Add(self.live_label, 0, wx.TOP | wx.LEFT | wx.RIGHT | wx.EXPAND, 8)
        
        # Multiline read-only TextCtrl is highly accessible and supports selection/screen reading
        self.live_text_ctrl = wx.TextCtrl(
            panel, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.TE_NOHIDESEL
        )
        self.live_text_ctrl.SetValue("Press F5 or click Start to begin captioning.")
        self.live_text_ctrl.SetToolTip("Real-time transcription preview.")
        main_sizer.Add(self.live_text_ctrl, 0, wx.ALL | wx.EXPAND, 8)
        
        # 3. Quick Control Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.start_btn = wx.Button(panel, label="Start Captioning (F5)")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        btn_sizer.Add(self.start_btn, 1, wx.ALL | wx.EXPAND, 4)
        
        self.stop_btn = wx.Button(panel, label="Stop Captioning (F6)")
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.stop_btn.Disable()
        btn_sizer.Add(self.stop_btn, 1, wx.ALL | wx.EXPAND, 4)
        
        self.clear_btn = wx.Button(panel, label="Clear History (Ctrl+L)")
        self.clear_btn.Bind(wx.EVT_BUTTON, self.on_clear)
        btn_sizer.Add(self.clear_btn, 1, wx.ALL | wx.EXPAND, 4)
        
        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, 4)
        
        # Set panel sizer
        panel.SetSizer(main_sizer)
        
        # Create Menu Bar
        self.create_menu()
        
        # Create Status Bar
        self.CreateStatusBar()
        self.update_status_bar()
        
        # Apply Font Settings
        self.apply_font_size()

    def create_menu(self):
        menubar = wx.MenuBar()
        
        # 1. File Menu
        file_menu = wx.Menu()
        clear_item = file_menu.Append(wx.ID_ANY, "Clear History\tCtrl+L", "Clear all transcript history")
        export_item = file_menu.Append(wx.ID_ANY, "Export Transcript...\tCtrl+E", "Export transcript to a text file")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "Exit", "Exit application")
        
        self.Bind(wx.EVT_MENU, self.on_clear, clear_item)
        self.Bind(wx.EVT_MENU, self.on_export, export_item)
        self.Bind(wx.EVT_MENU, lambda evt: self.Close(), exit_item)
        
        # 2. Source Menu (Audio Input sources)
        self.source_menu = wx.Menu()
        self.populate_sources()
        
        # 3. Model Menu (Whisper Settings)
        model_menu = wx.Menu()
        
        # Model Sizes
        self.model_items = {}
        for size in ["tiny", "base", "small", "medium"]:
            item = model_menu.AppendRadioItem(wx.ID_ANY, f"Model: {size.capitalize()}", f"Use Whisper {size} model")
            self.model_items[size] = item
            if size == self.current_model_size:
                item.Check(True)
            self.Bind(wx.EVT_MENU, self.on_change_model, item)
            
        model_menu.AppendSeparator()
        
        # Model Devices
        self.device_items = {}
        cuda_supported = is_cuda_available()
        for dev in ["cuda", "cpu"]:
            label = f"Device: GPU (CUDA)" if dev == "cuda" else "Device: CPU"
            item = model_menu.AppendRadioItem(wx.ID_ANY, label, f"Run transcription on {dev.upper()}")
            self.device_items[dev] = item
            if not cuda_supported and dev == "cuda":
                item.Enable(False)
            if dev == self.current_device:
                item.Check(True)
            self.Bind(wx.EVT_MENU, self.on_change_device, item)
            
        # 4. Options Menu (Accessibility and Calibration)
        options_menu = wx.Menu()
        
        # Font Sizes
        font_menu = wx.Menu()
        self.font_items = {}
        for name in FONT_SIZES.keys():
            item = font_menu.AppendRadioItem(wx.ID_ANY, name, f"Set text size to {name}")
            self.font_items[name] = item
            if name == self.font_size_name:
                item.Check(True)
            self.Bind(wx.EVT_MENU, self.on_change_font_size, item)
        options_menu.AppendSubMenu(font_menu, "Text Size")
        
        # Silence Threshold (Sensitivity)
        sens_menu = wx.Menu()
        self.sens_items = {}
        for name in SENSITIVITY_MAP.keys():
            item = sens_menu.AppendRadioItem(wx.ID_ANY, name, f"Set voice detection sensitivity to {name}")
            self.sens_items[name] = item
            if name == self.sensitivity:
                item.Check(True)
            self.Bind(wx.EVT_MENU, self.on_change_sensitivity, item)
        options_menu.AppendSubMenu(sens_menu, "Voice Sensitivity")
        
        # Silence Timeout
        timeout_menu = wx.Menu()
        self.timeout_items = {}
        for name in TIMEOUT_MAP.keys():
            item = timeout_menu.AppendRadioItem(wx.ID_ANY, name, f"Set phrase pause timeout to {name}")
            self.timeout_items[name] = item
            if name == self.silence_timeout:
                item.Check(True)
            self.Bind(wx.EVT_MENU, self.on_change_timeout, item)
        options_menu.AppendSubMenu(timeout_menu, "Pause Duration (Silence Timeout)")
        
        options_menu.AppendSeparator()
        
        # Auto-Read (TTS) Toggle
        tts_item = options_menu.AppendCheckItem(wx.ID_ANY, "Read Captions Automatically (TTS)", "Use SAPI5 to speak captions")
        tts_item.Check(self.tts_enabled)
        self.Bind(wx.EVT_MENU, self.on_toggle_tts, tts_item)
        
        # Add to MenuBar
        menubar.Append(file_menu, "File")
        menubar.Append(self.source_menu, "Audio Source")
        menubar.Append(model_menu, "Whisper Model")
        menubar.Append(options_menu, "Options")
        
        self.SetMenuBar(menubar)

    def populate_sources(self):
        # Clear existing items if any
        for item in list(self.source_menu.GetMenuItems()):
            self.source_menu.DestroyItem(item)
            
        p = pyaudio.PyAudio()
        wasapi_index = -1
        for i in range(p.get_host_api_count()):
            if p.get_host_api_info_by_index(i)['type'] == pyaudio.paWASAPI:
                wasapi_index = i
                break
                
        self.source_devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if wasapi_index != -1 and info['hostApi'] != wasapi_index:
                continue
            if info['maxInputChannels'] > 0:
                self.source_devices.append(info)
                
        p.terminate()
        
        if not self.source_devices:
            self.source_menu.Append(wx.ID_ANY, "No WASAPI input devices found").Enable(False)
            return
            
        # Add devices as radio items
        self.source_menu_items = {}
        for idx, dev in enumerate(self.source_devices):
            label = dev['name']
            if dev.get('isLoopbackDevice', False):
                label += " [System Audio Loopback]"
            else:
                label += " [Microphone]"
                
            item = self.source_menu.AppendRadioItem(wx.ID_ANY, f"{label}", f"Record from {dev['name']}")
            self.source_menu_items[dev['index']] = item
            
            # Default selection: prioritize loopback if available, else first microphone
            if self.selected_device_index == -1:
                if dev.get('isLoopbackDevice', False) or idx == 0:
                    self.selected_device_index = dev['index']
                    
            if dev['index'] == self.selected_device_index:
                item.Check(True)
                
            self.Bind(wx.EVT_MENU, lambda evt, d_idx=dev['index']: self.on_select_device(d_idx), item)

    def setup_shortcuts(self):
        # Bind accelerator keys
        start_id = wx.NewIdRef()
        stop_id = wx.NewIdRef()
        clear_id = wx.NewIdRef()
        export_id = wx.NewIdRef()
        
        self.Bind(wx.EVT_MENU, self.on_start, id=start_id)
        self.Bind(wx.EVT_MENU, self.on_stop, id=stop_id)
        self.Bind(wx.EVT_MENU, self.on_clear, id=clear_id)
        self.Bind(wx.EVT_MENU, self.on_export, id=export_id)
        
        accel_table = wx.AcceleratorTable([
            (wx.ACCEL_NORMAL, wx.WXK_F5, start_id),
            (wx.ACCEL_NORMAL, wx.WXK_F6, stop_id),
            (wx.ACCEL_CTRL, ord('L'), clear_id),
            (wx.ACCEL_CTRL, ord('E'), export_id)
        ])
        self.SetAcceleratorTable(accel_table)

    def apply_font_size(self):
        size = FONT_SIZES.get(self.font_size_name, 16)
        # Create an accessible, easy-to-read sans-serif font
        font = wx.Font(size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        
        # Apply to text controllers and listboxes
        self.listbox_label.SetFont(font)
        self.history_listbox.SetFont(font)
        self.live_label.SetFont(font)
        self.live_text_ctrl.SetFont(font)
        self.start_btn.SetFont(font)
        self.stop_btn.SetFont(font)
        self.clear_btn.SetFont(font)
        
        self.Layout()

    def update_status_bar(self, custom_msg=""):
        p = pyaudio.PyAudio()
        dev_name = "None Selected"
        for d in self.source_devices:
            if d['index'] == self.selected_device_index:
                dev_name = d['name']
                if d.get('isLoopbackDevice', False):
                    dev_name += " [Loopback]"
                break
        p.terminate()
        
        status_text = f"Source: {dev_name} | Model: {self.current_model_size} ({self.current_device.upper()}) | "
        if self.model_loading:
            status_text += "LOADING WHISPER MODEL..."
        elif self.recording_active:
            status_text += "LISTENING & TRANSCRIBING..."
        else:
            status_text += "Idle"
            
        if custom_msg:
            status_text += f" | {custom_msg}"
            
        self.SetStatusText(status_text)

    # ------------------ EVENT HANDLERS ------------------
    
    def on_select_device(self, idx):
        if self.recording_active:
            wx.MessageBox("Please stop captioning before switching audio devices.", "Info", wx.OK | wx.ICON_INFORMATION)
            # Recheck correct menu item
            self.source_menu_items[self.selected_device_index].Check(True)
            return
        self.selected_device_index = idx
        self.update_status_bar()

    def on_change_model(self, evt):
        if self.recording_active:
            wx.MessageBox("Please stop captioning before changing models.", "Info", wx.OK | wx.ICON_INFORMATION)
            self.model_items[self.current_model_size].Check(True)
            return
            
        for size, item in self.model_items.items():
            if item.IsChecked():
                self.current_model_size = size
                break
        self.trigger_model_load()

    def on_change_device(self, evt):
        if self.recording_active:
            wx.MessageBox("Please stop captioning before switching devices.", "Info", wx.OK | wx.ICON_INFORMATION)
            self.device_items[self.current_device].Check(True)
            return
            
        for dev, item in self.device_items.items():
            if item.IsChecked():
                self.current_device = dev
                break
        self.trigger_model_load()

    def on_change_font_size(self, evt):
        for name, item in self.font_items.items():
            if item.IsChecked():
                self.font_size_name = name
                break
        self.apply_font_size()

    def on_change_sensitivity(self, evt):
        for name, item in self.sens_items.items():
            if item.IsChecked():
                self.sensitivity = name
                break

    def on_change_timeout(self, evt):
        for name, item in self.timeout_items.items():
            if item.IsChecked():
                self.silence_timeout = name
                break

    def on_toggle_tts(self, evt):
        self.tts_enabled = evt.IsChecked()

    def on_start(self, evt):
        if self.model_loading:
            wx.MessageBox("Still loading Whisper model. Please wait.", "Warning", wx.OK | wx.ICON_WARNING)
            return
        if not self.whisper_model:
            wx.MessageBox("Whisper model is not loaded. Try reloading.", "Error", wx.OK | wx.ICON_ERROR)
            return
        if self.recording_active:
            return
            
        self.start_recording()

    def on_stop(self, evt):
        if not self.recording_active:
            return
        self.stop_recording()

    def on_clear(self, evt):
        self.history_listbox.Clear()
        self.live_text_ctrl.SetValue("")

    def on_export(self, evt):
        items = self.history_listbox.GetStrings()
        if not items:
            wx.MessageBox("No transcript to export.", "Info", wx.OK | wx.ICON_INFORMATION)
            return
            
        with wx.FileDialog(self, "Export Transcript", wildcard="Text files (*.txt)|*.txt",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            path = fileDialog.GetPath()
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(items))
                wx.MessageBox("Transcript exported successfully.", "Success", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"Failed to export transcript: {e}", "Error", wx.OK | wx.ICON_ERROR)

    # ------------------ WHISPER MODEL LOADING (BACKGROUND THREAD) ------------------
    
    def trigger_model_load(self):
        self.model_loading = True
        self.start_btn.Disable()
        self.update_status_bar()
        
        # Start loading model in background thread
        thread = threading.Thread(target=self._bg_load_model, daemon=True)
        thread.start()

    def _bg_load_model(self):
        size = self.current_model_size
        device = self.current_device
        # Use fp16 for CUDA (fastest), int8 for CPU (highly optimized)
        comp_type = "float16" if device == "cuda" else "int8"
        
        try:
            # Clean up old model to free RAM/VRAM
            if self.whisper_model:
                del self.whisper_model
                self.whisper_model = None
                
            model = WhisperModel(size, device=device, compute_type=comp_type)
            wx.CallAfter(self._on_model_loaded, model)
        except Exception as e:
            wx.CallAfter(self._on_model_load_failed, str(e))

    def _on_model_loaded(self, model):
        self.whisper_model = model
        self.model_loading = False
        self.start_btn.Enable()
        self.update_status_bar("Model loaded successfully!")
        
        # Reset live text status message
        self.live_text_ctrl.SetValue("Model ready. Press F5 or click Start to begin.")

    def _on_model_load_failed(self, err_msg):
        self.whisper_model = None
        self.model_loading = False
        self.update_status_bar("MODEL LOAD FAILED!")
        wx.MessageBox(f"Failed to load Whisper model:\n{err_msg}\n\nFalling back to CPU might help.", "Error", wx.OK | wx.ICON_ERROR)
        self.live_text_ctrl.SetValue("Model load failed. Choose CPU or another model size and try again.")

    # ------------------ RECORDING & PIPELINE ------------------
    
    def start_recording(self):
        self.recording_active = True
        self.running = True
        self.start_btn.Disable()
        self.stop_btn.Enable()
        self.update_status_bar()
        
        self.live_text_ctrl.SetValue("Initializing audio stream...")
        
        # Reset active audio buffers
        self.active_speech_buffer = []
        self.silence_samples_count = 0
        self.last_live_transcribe_time = time.time()
        
        # Clear queues
        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.transcription_queue.empty():
            self.transcription_queue.get()
            
        # Start helper threads
        self.pipeline_thread = threading.Thread(target=self._audio_pipeline_loop, daemon=True)
        self.transcription_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        
        self.pipeline_thread.start()
        self.transcription_thread.start()
        
        # Initialize Audio Device Stream
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            # Query device settings
            dev_info = self.pyaudio_instance.get_device_info_by_index(self.selected_device_index)
            self.native_rate = int(dev_info['defaultSampleRate'])
            self.native_channels = dev_info['maxInputChannels']
            
            def audio_callback(in_data, frame_count, time_info, status):
                if self.recording_active:
                    self.audio_queue.put(in_data)
                return (None, pyaudio.paContinue)
                
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.native_channels,
                rate=self.native_rate,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=1024,
                stream_callback=audio_callback
            )
            self.live_text_ctrl.SetValue("Listening...")
            self.update_status_bar()
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            self.stop_recording()
            wx.MessageBox(f"Failed to start audio stream:\n{e}", "Error", wx.OK | wx.ICON_ERROR)

    def stop_recording(self):
        self.recording_active = False
        self.running = False
        
        # Close audio stream safely
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None
            
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
            
        self.start_btn.Enable()
        self.stop_btn.Disable()
        self.update_status_bar()
        self.live_text_ctrl.SetValue("Stopped.")

    def _audio_pipeline_loop(self):
        """Runs in background: Consumes raw bytes, downmixes, resamples, does VAD, and schedules transcription."""
        while self.running:
            try:
                # Blocks with timeout to keep thread alive checks
                raw_bytes = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                # Periodically check if there is un-transcribed audio in the active buffer to perform a live update
                if self.active_speech_buffer and (time.time() - self.last_live_transcribe_time > self.live_update_interval):
                    # Request a live (partial) update
                    audio_data = np.concatenate(self.active_speech_buffer)
                    self.transcription_queue.put((audio_data, False)) # False = live/partial caption
                    self.last_live_transcribe_time = time.time()
                continue
                
            # Convert bytes to numpy float32
            chunk = np.frombuffer(raw_bytes, dtype=np.float32)
            if len(chunk) == 0:
                continue
                
            # Downmix to mono
            if self.native_channels > 1:
                chunk = chunk.reshape(-1, self.native_channels).mean(axis=1)
                
            # Resample to 16000Hz (Whisper native sample rate)
            resampled = self.resample_chunk(chunk, self.native_rate, 16000)
            
            # VAD Processing
            threshold = SENSITIVITY_MAP.get(self.sensitivity, 0.006)
            timeout = TIMEOUT_MAP.get(self.silence_timeout, 1.5)
            
            # Calculate RMS energy of this block
            rms = np.sqrt(np.mean(resampled ** 2))
            
            if rms > threshold:
                # Active speech
                self.active_speech_buffer.append(resampled)
                self.silence_samples_count = 0
            else:
                # Silent block
                if self.active_speech_buffer:
                    self.active_speech_buffer.append(resampled)
                    self.silence_samples_count += len(resampled)
                    
                    # Check if silence timeout exceeded
                    silence_seconds = self.silence_samples_count / 16000
                    if silence_seconds >= timeout:
                        # Finalize phrase
                        audio_data = np.concatenate(self.active_speech_buffer)
                        self.transcription_queue.put((audio_data, True)) # True = final caption
                        
                        # Reset buffers
                        self.active_speech_buffer = []
                        self.silence_samples_count = 0
                        
            # Force finalize if the phrase is too long (avoid infinite buffering and latency)
            if self.active_speech_buffer:
                buffer_duration = sum(len(x) for x in self.active_speech_buffer) / 16000
                if buffer_duration > 15.0: # Max 15s segments
                    audio_data = np.concatenate(self.active_speech_buffer)
                    self.transcription_queue.put((audio_data, True))
                    
                    # Keep the last 0.5s to avoid clipping boundary words
                    overlap_samples = int(0.5 * 16000)
                    if len(audio_data) > overlap_samples:
                        self.active_speech_buffer = [audio_data[-overlap_samples:]]
                    else:
                        self.active_speech_buffer = []
                    self.silence_samples_count = 0

    def resample_chunk(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        orig_indices = np.arange(len(audio))
        target_indices = np.linspace(0, len(audio) - 1, int(duration * target_sr))
        return np.interp(target_indices, orig_indices, audio).astype(np.float32)

    # ------------------ TRANSCRIPTION WORKER (BACKGROUND THREAD) ------------------
    
    def _transcription_loop(self):
        """Runs in background: Consumes audio matrices, runs Whisper inference, and updates the GUI."""
        while self.running:
            try:
                audio_data, is_final = self.transcription_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if not self.whisper_model:
                continue
                
            try:
                # Transcribe with VAD filter enabled for safety
                segments, _ = self.whisper_model.transcribe(audio_data, beam_size=2, vad_filter=True)
                text = " ".join([segment.text for segment in segments]).strip()
                
                # Filter out garbage repetitions or empty transcribes
                if not text or text.lower() in [".", "you", "thank you", "thank you.", "bye", "bye."]:
                    continue
                    
                if is_final:
                    # Finalized phrase - add to History ListBox and read aloud if TTS enabled
                    wx.CallAfter(self.add_final_caption, text)
                else:
                    # Live/partial update - show in live preview box (italicized or highlighted)
                    wx.CallAfter(self.update_live_caption, text)
            except Exception as e:
                print(f"Transcription error: {e}")

    def update_live_caption(self, text):
        # Format live text nicely (e.g. adding brackets)
        self.live_text_ctrl.SetValue(f"... {text}")

    def add_final_caption(self, text):
        # Reset live preview box
        self.live_text_ctrl.SetValue("")
        
        # Append to History ListBox (accessible native control)
        index = self.history_listbox.Append(text)
        self.history_listbox.EnsureVisible(index)
        
        # Screen reader accessibility alert (if needed, NVDA handles ListBox updates well when focused)
        # But we can also set the tooltip or label dynamically to raise focus events
        
        # Run TTS narration in a separate thread if enabled (abiding by SAPI5 isolation rule)
        if self.tts_enabled:
            threading.Thread(target=self._run_tts, args=(text,), daemon=True).start()

    def _run_tts(self, text):
        # Abide by the safety convention: Initialize fresh engine instance per segment to avoid SAPI5 driver hangs
        try:
            engine = pyttsx3.init()
            # Set speaking rate to a clear pace
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            # Clean up object reference
            del engine
        except Exception as e:
            print(f"TTS offline speech failure: {e}")

    def on_close(self, evt):
        self.stop_recording()
        self.Destroy()

class CaptionApp(wx.App):
    def OnInit(self):
        self.frame = CaptionFrame(None, title="Accessible Live Captions")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = CaptionApp()
    app.MainLoop()
