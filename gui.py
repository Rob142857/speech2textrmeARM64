#!/usr/bin/env python3
"""
GUI Interface for ARM64 Whisper Transcription
Professional tkinter interface optimized for Windows ARM64
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import time
from typing import Dict, Any, Optional
import queue
import json

# Production transcriber import (placeholder removed)
try:
    # Prefer temporary transcriber variant if present (development mode)
    from transcribe_temp import ARM64WhisperTranscriber  # type: ignore
    TRANSCRIBER_AVAILABLE = True
    print("✅ Using temporary transcriber (dev mode)")
except ImportError:
    try:
        from transcribe import ARM64WhisperTranscriber  # type: ignore
        TRANSCRIBER_AVAILABLE = True
        print("✅ Using full transcriber")
    except ImportError as e:
        print(f"⚠️  Transcriber not available: {e}")
        TRANSCRIBER_AVAILABLE = False

class WhisperGUI:
    """
    Professional GUI for ARM64 Whisper transcription with real-time progress.
    """
    
    def __init__(self, root: tk.Tk):
        """Initialize the GUI application."""
        print("🎨 Initializing GUI components...")
        self.root = root
        self.transcriber = None
        self.progress_queue = queue.Queue()
        self.is_transcribing = False
        self.current_thread = None
        
        # Setup window
        print("🏠 Setting up window...")
        self.setup_window()
        
        # Create interface
        print("🧩 Creating widgets...")
        self.create_widgets()
        
        # Initialize transcriber if available
        if TRANSCRIBER_AVAILABLE:
            self.initialize_transcriber()
        else:
            self.show_transcriber_error()
        
        # Start progress monitoring
        self.root.after(100, self.check_progress_queue)
    
    def setup_window(self):
        """Setup main window properties."""
        self.root.title("ARM64 Whisper Transcription - NPU Accelerated")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Set icon if available
        try:
            # You can add an icon file here
            # self.root.iconbitmap("icon.ico")
            pass
        except:
            pass
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')  # Modern looking theme
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)  # Results area
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="ARM64 Whisper Transcription", 
            font=("Segoe UI", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        # System info
        self.create_system_info_frame(main_frame)
        
        # Input section
        self.create_input_section(main_frame)
        
        # Options section
        self.create_options_section(main_frame)
        
        # Progress section
        self.create_progress_section(main_frame)
        
        # Results section
        self.create_results_section(main_frame)
        
        # Action buttons
        self.create_action_buttons(main_frame)
    
    def create_system_info_frame(self, parent):
        """Create system information display."""
        info_frame = ttk.LabelFrame(parent, text="System Information", padding="5")
        info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # System info display
        self.system_info_var = tk.StringVar()
        self.system_info_var.set("Initializing system information...")
        
        info_label = ttk.Label(info_frame, textvariable=self.system_info_var)
        info_label.grid(row=0, column=0, sticky="w")
        
        # NPU status indicator
        self.npu_status_var = tk.StringVar()
        self.npu_status_var.set("NPU Status: Checking...")
        
        npu_label = ttk.Label(info_frame, textvariable=self.npu_status_var)
        npu_label.grid(row=1, column=0, sticky="w")
    
    def create_input_section(self, parent):
        """Create file input section."""
        input_frame = ttk.LabelFrame(parent, text="Input File", padding="5")
        input_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        input_frame.grid_columnconfigure(1, weight=1)
        
        # File selection
        ttk.Label(input_frame, text="Audio/Video File:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(input_frame, textvariable=self.file_path_var, state="readonly")
        file_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        
        browse_button = ttk.Button(input_frame, text="Browse...", command=self.browse_file)
        browse_button.grid(row=0, column=2, sticky="e")
        
        # File info display
        self.file_info_var = tk.StringVar()
        info_label = ttk.Label(input_frame, textvariable=self.file_info_var, foreground="gray")
        info_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))
    
    def create_options_section(self, parent):
        """Create transcription options section."""
        options_frame = ttk.LabelFrame(parent, text="Transcription Options", padding="5")
        options_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        options_frame.grid_columnconfigure(1, weight=1)
        
        # Model selection
        ttk.Label(options_frame, text="Whisper Model:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.model_var = tk.StringVar(value="large")
        model_combo = ttk.Combobox(
            options_frame, 
            textvariable=self.model_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly"
        )
        model_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        
        # Language selection
        ttk.Label(options_frame, text="Language:").grid(row=0, column=2, sticky="w", padx=(10, 5))
        self.language_var = tk.StringVar(value="auto")
        language_combo = ttk.Combobox(
            options_frame,
            textvariable=self.language_var,
            values=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            state="readonly"
        )
        language_combo.grid(row=0, column=3, sticky="w")
        
        # Output directory
        ttk.Label(options_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "output"))
        output_entry = ttk.Entry(options_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 5), pady=(10, 0))
        
        output_browse_button = ttk.Button(options_frame, text="Browse...", command=self.browse_output_dir)
        output_browse_button.grid(row=1, column=3, sticky="e", pady=(10, 0))
        
        # Advanced options
        advanced_frame = ttk.Frame(options_frame)
        advanced_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        
        self.use_npu_var = tk.BooleanVar(value=True)
        npu_check = ttk.Checkbutton(advanced_frame, text="Use NPU Acceleration", variable=self.use_npu_var)
        npu_check.grid(row=0, column=0, sticky="w")
        
        self.word_timestamps_var = tk.BooleanVar(value=False)
        timestamps_check = ttk.Checkbutton(advanced_frame, text="Word Timestamps", variable=self.word_timestamps_var)
        timestamps_check.grid(row=0, column=1, sticky="w", padx=(20, 0))
        
        self.verbose_var = tk.BooleanVar(value=True)
        verbose_check = ttk.Checkbutton(advanced_frame, text="Verbose Output", variable=self.verbose_var)
        verbose_check.grid(row=0, column=2, sticky="w", padx=(20, 0))
    
    def create_progress_section(self, parent):
        """Create progress monitoring section."""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        progress_frame.grid_columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            mode="determinate",
            length=400
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to transcribe")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, sticky="w")
    
    def create_results_section(self, parent):
        """Create results display section."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="5")
        results_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            height=12,
            font=("Consolas", 10)
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        # Context menu for results
        self.create_results_context_menu()
    
    def create_results_context_menu(self):
        """Create context menu for results text area."""
        self.results_menu = tk.Menu(self.root, tearoff=0)
        self.results_menu.add_command(label="Select All", command=self.select_all_results)
        self.results_menu.add_command(label="Copy", command=self.copy_results)
        self.results_menu.add_separator()
        self.results_menu.add_command(label="Save As...", command=self.save_results_as)
        self.results_menu.add_command(label="Clear", command=self.clear_results)
        
        self.results_text.bind("<Button-3>", self.show_results_menu)
    
    def create_action_buttons(self, parent):
        """Create action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=6, column=0, columnspan=2, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        
        # Left side - info button
        info_button = ttk.Button(button_frame, text="System Info", command=self.show_system_info)
        info_button.grid(row=0, column=0, sticky="w")
        
        # Right side - action buttons
        action_frame = ttk.Frame(button_frame)
        action_frame.grid(row=0, column=1, sticky="e")
        
        self.transcribe_button = ttk.Button(
            action_frame,
            text="Start Transcription",
            command=self.start_transcription,
            style="Accent.TButton"
        )
        self.transcribe_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(
            action_frame,
            text="Stop",
            command=self.stop_transcription,
            state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        
        clear_button = ttk.Button(action_frame, text="Clear Results", command=self.clear_results)
        clear_button.grid(row=0, column=2)
    
    def initialize_transcriber(self):
        """Initialize the transcription engine."""
        try:
            self.transcriber = ARM64WhisperTranscriber()
            # Detect native accelerated path
            native_dir = Path.cwd() / 'native'
            native_available = False
            if native_dir.exists():
                try:
                    if str(native_dir) not in sys.path:
                        sys.path.insert(0, str(native_dir))
                    import importlib
                    importlib.import_module('tokenizer_native')
                    importlib.import_module('mel_native')
                    native_available = True
                except Exception:
                    native_available = False
            
            # Update system info
            system_info = f"Platform: {self.transcriber.system_info['platform']} {self.transcriber.system_info['architecture']}"
            self.system_info_var.set(system_info)
            
            # Update NPU status
            npu_info = self.transcriber.whisper_npu.get_npu_info() if self.transcriber.whisper_npu else {}
            if npu_info.get("npu_available", False):
                npu_status = f"NPU Status: Available ({', '.join(npu_info['providers'])})"
                self.npu_status_var.set(npu_status)
            else:
                self.npu_status_var.set("NPU Status: Not Available (CPU Mode)")
            
            # Append processing mode
            mode = 'Native ONNX/QNN' if native_available else 'Python'
            self.npu_status_var.set(self.npu_status_var.get() + f" | Mode: {mode}")

            self.log_message("✅ ARM64 Whisper transcriber initialized successfully")
            
        except Exception as e:
            self.show_transcriber_error(str(e))
    
    def show_transcriber_error(self, error: str = None):
        """Show transcriber initialization error."""
        error_msg = f"Transcriber initialization failed: {error}" if error else "Transcriber not available"
        self.system_info_var.set("❌ Transcriber Error")
        self.npu_status_var.set("NPU Status: Unavailable")
        self.log_message(f"❌ {error_msg}")
        self.transcribe_button.config(state="disabled")
    
    def browse_file(self):
        """Browse for input file."""
        filetypes = [
            ("All Supported", "*.mp3;*.mp4;*.wav;*.m4a;*.avi;*.mov;*.mkv;*.webm"),
            ("Audio Files", "*.mp3;*.wav;*.m4a;*.aac;*.ogg;*.flac"),
            ("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.webm;*.wmv"),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio or Video File",
            filetypes=filetypes,
            initialdir=str(Path.home())
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.update_file_info(filename)
    
    def update_file_info(self, filepath: str):
        """Update file information display."""
        try:
            path = Path(filepath)
            size = path.stat().st_size
            size_mb = size / (1024 * 1024)
            
            info = f"File: {path.name} ({size_mb:.1f} MB)"
            
            # Get additional info if transcriber available
            if self.transcriber and hasattr(self.transcriber, 'audio_processor'):
                audio_info = self.transcriber.audio_processor.get_audio_info(filepath)
                if audio_info:
                    duration = audio_info.get('duration', 0)
                    info += f" - Duration: {duration:.1f}s"
            
            self.file_info_var.set(info)
            
        except Exception as e:
            self.file_info_var.set(f"Error reading file info: {e}")
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        
        if directory:
            self.output_dir_var.set(directory)
    
    def start_transcription(self):
        """Start transcription in background thread."""
        if not self.transcriber:
            messagebox.showerror("Error", "Transcriber not initialized")
            return
        
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not Path(self.file_path_var.get()).exists():
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        # Disable controls
        self.is_transcribing = True
        self.transcribe_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Clear previous results
        self.clear_results()
        
        # Start transcription thread
        self.current_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        self.current_thread.start()
    
    def transcription_worker(self):
        """Background transcription worker."""
        try:
            self.progress_queue.put(("status", "Starting transcription..."))
            self.progress_queue.put(("progress", 10))
            
            # Prepare options
            language = None if self.language_var.get() == "auto" else self.language_var.get()
            
            transcription_options = {
                "language": language,
                "temperature": 0.0,
                "word_timestamps": self.word_timestamps_var.get(),
                "verbose": self.verbose_var.get()
            }
            
            # Update model if needed
            if hasattr(self.transcriber, 'model_name') and self.transcriber.model_name != self.model_var.get():
                self.progress_queue.put(("status", f"Loading model: {self.model_var.get()}"))
                # In a full implementation, we'd reload the model here
            
            # Update NPU setting
            if self.transcriber.whisper_npu:
                self.transcriber.whisper_npu.use_npu = self.use_npu_var.get()
            
            self.progress_queue.put(("progress", 30))
            
            # Decide backend: native if artifacts present
            native_dir = Path.cwd() / 'native'
            model_guess = Path.cwd() / 'models'  # naive heuristics
            onnx_dirs = list((model_guess).glob('**/encoder.onnx'))
            # Prefer large model if multiple
            if len(onnx_dirs) > 1:
                def pref(p):
                    name = p.parent.name.lower()
                    return (0 if 'large' in name else 1, -p.stat().st_size)
                onnx_dirs.sort(key=pref)
            native_used = False
            if native_dir.exists():
                try:
                    from native_backend import native_transcribe  # type: ignore
                    # choose first encoder.onnx parent as model dir
                    model_dir = onnx_dirs[0].parent if onnx_dirs else None
                    if model_dir:
                        self.progress_queue.put(("status", f"Native QNN decode ({model_dir.name})"))
                        native_res = native_transcribe(str(model_dir), self.file_path_var.get(), language=self.language_var.get() if self.language_var.get()!='auto' else None)
                        result = {
                            'success': True,
                            'transcription': native_res['text'],
                            'metadata': native_res,
                            'output_files': {},
                            'processing_time': native_res['timings']['encoder_s'] + native_res['timings']['decode_s']
                        }
                        native_used = True
                    else:
                        result = self.transcriber.transcribe_file(
                            input_path=self.file_path_var.get(),
                            **transcription_options
                        )
                except Exception:
                    result = self.transcriber.transcribe_file(
                        input_path=self.file_path_var.get(),
                        **transcription_options
                    )
            else:
                result = self.transcriber.transcribe_file(
                    input_path=self.file_path_var.get(),
                    **transcription_options
                )
            
            if result.get("success", False):
                self.progress_queue.put(("progress", 100))
                self.progress_queue.put(("status", "Transcription completed successfully"))
                self.progress_queue.put(("result", result))
            else:
                self.progress_queue.put(("error", f"Transcription failed: {result.get('error', 'Unknown error')}"))
            
        except Exception as e:
            self.progress_queue.put(("error", f"Transcription error: {str(e)}"))
        finally:
            self.progress_queue.put(("done", None))
    
    def stop_transcription(self):
        """Stop current transcription."""
        if self.current_thread and self.current_thread.is_alive():
            # Note: Python threads can't be forcibly stopped
            # This is more of a UI update
            self.is_transcribing = False
            self.status_var.set("Stopping transcription...")
            self.log_message("⏹️ Transcription stop requested")
    
    def check_progress_queue(self):
        """Check progress queue for updates."""
        try:
            while True:
                msg_type, data = self.progress_queue.get_nowait()
                
                if msg_type == "status":
                    self.status_var.set(data)
                    
                elif msg_type == "progress":
                    self.progress_var.set(data)
                    
                elif msg_type == "result":
                    self.display_results(data)
                    
                elif msg_type == "error":
                    self.status_var.set("Error occurred")
                    self.log_message(f"❌ {data}")
                    messagebox.showerror("Transcription Error", data)
                    
                elif msg_type == "done":
                    self.transcription_finished()
                    
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.check_progress_queue)
    
    def transcription_finished(self):
        """Handle transcription completion."""
        self.is_transcribing = False
        self.transcribe_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_var.set(0)
    
    def display_results(self, result: Dict[str, Any]):
        """Display transcription results."""
        if not result.get("success", False):
            self.log_message("❌ Transcription failed")
            return
        
        # Display transcription text
        transcription = result.get("transcription", "")
        if transcription:
            self.results_text.insert(tk.END, "TRANSCRIPTION RESULT\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n\n")
            self.results_text.insert(tk.END, transcription)
            self.results_text.insert(tk.END, "\n\n" + "=" * 50 + "\n")
        
        # Display metadata
        metadata = result.get("metadata", {})
        if metadata:
            self.results_text.insert(tk.END, "\nTRANSCRIPTION DETAILS\n")
            self.results_text.insert(tk.END, "-" * 25 + "\n")
            
            for key, value in metadata.items():
                if key not in ["input_file", "output_files"]:
                    display_key = key.replace('_', ' ').title()
                    self.results_text.insert(tk.END, f"{display_key}: {value}\n")
        
        # Display output file info
        output_files = result.get("output_files", {})
        if output_files:
            self.results_text.insert(tk.END, "\nOUTPUT FILES\n")
            self.results_text.insert(tk.END, "-" * 12 + "\n")
            for file_type, file_path in output_files.items():
                if file_path:
                    self.results_text.insert(tk.END, f"{file_type.upper()}: {file_path}\n")
        
        # Scroll to top
        self.results_text.see("1.0")
        
        # Log completion
        processing_time = result.get("processing_time", 0)
        self.log_message(f"✅ Transcription completed in {processing_time:.1f} seconds")
    
    def log_message(self, message: str):
        """Add a message to the results area."""
        timestamp = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
    
    def clear_results(self):
        """Clear results text area."""
        self.results_text.delete(1.0, tk.END)
    
    def select_all_results(self):
        """Select all text in results area."""
    self.results_text.tag_add(tk.SEL, "1.0", tk.END)
    self.results_text.mark_set(tk.INSERT, "1.0")
    self.results_text.see("1.0")
    
    def copy_results(self):
        """Copy selected text to clipboard."""
        try:
            selected_text = self.results_text.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
        except tk.TclError:
            # No text selected, copy all
            all_text = self.results_text.get("1.0", tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(all_text)
    
    def save_results_as(self):
        """Save results to file."""
        content = self.results_text.get("1.0", tk.END)
        if not content.strip():
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_message(f"✅ Results saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def show_results_menu(self, event):
        """Show context menu for results area."""
        self.results_menu.post(event.x_root, event.y_root)
    
    def show_system_info(self):
        """Show detailed system information."""
        info_window = tk.Toplevel(self.root)
        info_window.title("System Information")
        info_window.geometry("600x400")
        info_window.resizable(True, True)
        
        # Create text area
        text_area = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, font=("Consolas", 10))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gather system information
        info_lines = [
            "ARM64 WHISPER TRANSCRIPTION SYSTEM INFORMATION",
            "=" * 55,
            "",
            f"Transcriber Available: {'Yes' if self.transcriber else 'No'}",
            ""
        ]
        
        if self.transcriber:
            info_lines.extend([
                "SYSTEM INFO:",
                f"  Platform: {self.transcriber.system_info['platform']}",
                f"  Architecture: {self.transcriber.system_info['architecture']}",
                f"  Python Version: {self.transcriber.system_info['python_version']}",
                f"  Is ARM64: {self.transcriber.system_info['is_arm64']}",
                ""
            ])
            
            # NPU information
            if self.transcriber.whisper_npu:
                npu_info = self.transcriber.whisper_npu.get_npu_info()
                info_lines.extend([
                    "NPU INFORMATION:",
                    f"  NPU Available: {npu_info.get('npu_available', False)}",
                    f"  Providers: {', '.join(npu_info.get('providers', []))}"
                ])
                
                if 'onnxruntime_version' in npu_info:
                    info_lines.append(f"  ONNX Runtime Version: {npu_info['onnxruntime_version']}")
                
                if 'available_providers' in npu_info:
                    info_lines.extend([
                        "",
                        "AVAILABLE ONNX PROVIDERS:",
                        "  " + "\n  ".join(npu_info['available_providers'])
                    ])
        
        # Insert text
        text_area.insert("1.0", "\n".join(info_lines))
        text_area.config(state="disabled")

def main():
    """Run the GUI application."""
    print("🎨 Starting GUI application...")
    
    # Create root window
    root = tk.Tk()
    print("✅ Tkinter root window created")
    
    # Create application
    print("🔧 Creating GUI application...")
    app = WhisperGUI(root)
    print("✅ GUI application created")
    
    # Handle window closing
    def on_closing():
        if app.is_transcribing:
            if messagebox.askokcancel("Quit", "Transcription is in progress. Do you want to quit?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    print("🚀 Starting GUI main loop...")
    try:
        root.mainloop()
        print("✅ GUI main loop ended normally")
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
