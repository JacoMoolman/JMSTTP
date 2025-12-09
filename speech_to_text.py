# Standard library imports
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from threading import Thread

# Third-party imports (imported after dependency check)
keyboard = None
sd = None
np = None
whisper = None
pyperclip = None
wavio = None
wavfile = None
pygame = None

def install_package(package_name, import_name=None):
    """Install a package using pip if it's not already installed."""
    if import_name is None:
        import_name = package_name
    
    # Check if package is already installed
    if importlib.util.find_spec(import_name) is not None:
        return True
    
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")
        return False

def check_whisper_conflict():
    """Check for whisper.py file conflicts in current directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    whisper_py = os.path.join(current_dir, 'whisper.py')
    
    if os.path.exists(whisper_py):
        print(f"ERROR: Found conflicting file: {whisper_py}")
        print("This file is interfering with the OpenAI Whisper import.")
        print("Please rename or remove the whisper.py file in your current directory.")
        return False
    return True

def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    dependencies = [
        ("keyboard", "keyboard"),
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("openai-whisper", "whisper"),
        ("pyperclip", "pyperclip"),
        ("wavio", "wavio"),
        ("scipy", "scipy"),
        ("pygame", "pygame")
    ]
    
    failed_installs = []
    for package_name, import_name in dependencies:
        if not install_package(package_name, import_name):
            failed_installs.append(package_name)
    
    if failed_installs:
        print(f"Failed to install the following packages: {', '.join(failed_installs)}")
        print("Please install them manually using: pip install " + " ".join(failed_installs))
        return False
    
    return True

# Check for file conflicts first
if not check_whisper_conflict():
    sys.exit(1)

# Ensure dependencies are installed before importing
if not ensure_dependencies():
    sys.exit(1)

# Now import the third-party modules after dependency check
try:
    import keyboard
    import sounddevice as sd
    import numpy as np
    import whisper
    import pyperclip
    import wavio
    from scipy.io import wavfile
    import pygame
except ImportError as e:
    print(f"Import error: {e}")
    print("There might be a package naming conflict or installation issue.")
    print("Try running: pip uninstall whisper && pip install openai-whisper")
    sys.exit(1)

def play_mp3(mp3_file):
    """Play an MP3 file with error handling (non-blocking)."""
    try:
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Get the full path to the MP3 file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mp3_path = os.path.join(script_dir, mp3_file)
        
        # Check if file exists
        if not os.path.exists(mp3_path):
            print(f"Warning: Sound file not found: {mp3_path}")
            return
        
        # Load and play the sound (non-blocking)
        sound = pygame.mixer.Sound(mp3_path)
        sound.play()
    except Exception as e:
        # If sound fails, don't break the main functionality
        print(f"Sound notification failed: {e}")

class SpeechToTextTranscriber:
    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model("small")
        print("Model loaded successfully!")
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.wav_filename = "latest_recording.wav"
        
    def record_audio(self):
        self.recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f'Error: {status}')
            if self.recording:
                self.audio_data.extend(indata[:, 0])
        
        try:
            with sd.InputStream(callback=callback, 
                              channels=1, 
                              samplerate=self.sample_rate,
                              dtype=np.float32):
                print("Recording... Press Ctrl+\" again to stop.")
                while self.recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error during recording: {e}")
    
    def stop_recording(self):
        self.recording = False
        
    def transcribe_audio(self):
        if not self.audio_data:
            print("No audio data recorded")
            return ""
            
        try:
            # Convert to numpy array
            audio_data = np.array(self.audio_data, dtype=np.float32)
            
            # Print some debug info
            print(f"Audio data shape: {audio_data.shape}")
            print(f"Audio data range: {np.min(audio_data)} to {np.max(audio_data)}")
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize audio to [-1, 1]
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Delete previous recording if it exists
            if os.path.exists(self.wav_filename):
                os.remove(self.wav_filename)
            
            # Save the WAV file
            wavfile.write(self.wav_filename, self.sample_rate, audio_data)
            print(f"Saved audio to {self.wav_filename}")
                
            # Transcribe using Whisper
            result = self.model.transcribe(audio_data)
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

def main():
    print("Initializing Speech-to-Text Transcriber...")
    try:
        transcriber = SpeechToTextTranscriber()
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        print("Make sure you have a working microphone and audio drivers installed.")
        return
    
    recording_thread = None
    
    def on_hotkey():
        nonlocal recording_thread
        
        if not transcriber.recording:
            # Start recording
            print("\nStarting new recording...")
            play_mp3("menu-button-88360.mp3")  # Sound notification: recording started
            recording_thread = Thread(target=transcriber.record_audio)
            recording_thread.start()
        else:
            # Stop recording and transcribe
            print("\nStopping recording...")
            transcriber.stop_recording()
            recording_thread.join()
            
            # Play sound to indicate processing has started
            play_mp3("ui_sci-fi-sound-36061.mp3")  # Sound notification: processing begins
            
            # Get transcription and copy to clipboard
            print("Transcribing audio...")
            text = transcriber.transcribe_audio()
            if text:
                print(f"Transcribed text: {text}")
                pyperclip.copy(text)
                time.sleep(0.1)  # Small delay to ensure clipboard is ready
                keyboard.write(text)
                print("Text has been pasted!")
            else:
                print("No text was transcribed")

    # Register the hotkey (Ctrl+")
    keyboard.add_hotkey('ctrl+"', on_hotkey)
    print("\n" + "="*50)
    print("Speech-to-Text is running!")
    print("Press Ctrl+\" to start/stop recording.")
    print("Press Esc to quit the program.")
    print("="*50)
    
    # Keep the program running
    try:
        keyboard.wait('esc')
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("Cleaning up...")
        if transcriber.recording:
            transcriber.stop_recording()
        if recording_thread and recording_thread.is_alive():
            recording_thread.join()

if __name__ == "__main__":
    main()