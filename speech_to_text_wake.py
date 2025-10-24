import subprocess
import sys
import importlib.util
import os


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
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("openai-whisper", "whisper"),
        ("pyperclip", "pyperclip"),
        ("wavio", "wavio"),
        ("scipy", "scipy"),
        ("pvporcupine", "pvporcupine"),
        ("keyboard", "keyboard")
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

# Now import the remaining required modules
try:
    import sounddevice as sd
    import numpy as np
    import whisper
    import pyperclip
    import time
    from scipy.io import wavfile
    import pvporcupine
    import keyboard
except ImportError as e:
    print(f"Import error: {e}")
    print("There might be a package naming conflict or installation issue.")
    sys.exit(1)

class WakeWordTranscriber:
    def __init__(self, access_key, wake_word, recording_duration=5):
        """
        Initialize with Porcupine access key.
        Get free key from: https://console.picovoice.ai/

        Built-in keywords: alexa, americano, blueberry, bumblebee, computer,
        grapefruit, grasshopper, hey google, hey siri, jarvis, ok google,
        picovoice, porcupine, terminator
        """
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("Model loaded successfully!")

        print(f"Initializing Porcupine with wake word: '{wake_word}'...")
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[wake_word]
            )
        except Exception as e:
            print(f"Error initializing Porcupine: {e}")
            print("\nMake sure you:")
            print("1. Have a valid access key from https://console.picovoice.ai/")
            print("2. Are using a valid built-in keyword")
            raise

        self.wake_word = wake_word
        self.recording_duration = recording_duration
        self.sample_rate = 16000
        self.porcupine_sample_rate = self.porcupine.sample_rate
        self.frame_length = self.porcupine.frame_length
        self.wav_filename = "latest_recording.wav"
        self.running = True

    def record_audio_for_duration(self, duration):
        """Record audio for a specific duration."""
        print(f"🎤 Recording for {duration} seconds...")
        audio_data = []

        def callback(indata, frames, time_info, status):
            if status:
                print(f'Status: {status}')
            audio_data.extend(indata[:, 0])

        with sd.InputStream(callback=callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          dtype=np.float32):
            time.sleep(duration)

        return np.array(audio_data, dtype=np.float32)

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper."""
        if len(audio_data) == 0:
            print("No audio data recorded")
            return ""

        try:
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
            print("Transcribing...")
            result = self.model.transcribe(audio_data)
            return result["text"].strip()

        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def listen_for_wake_word(self):
        """Continuously listen for wake word."""
        print(f"\n{'='*60}")
        print(f"👂 Listening for wake word: '{self.wake_word}'")
        print(f"After detection, will record for {self.recording_duration} seconds")
        print("Press Esc to quit")
        print(f"{'='*60}\n")

        audio_buffer = []

        def callback(indata, frames, time_info, status):
            if status:
                print(f'Status: {status}')
            # Convert float32 to int16 for Porcupine
            int16_data = (indata[:, 0] * 32767).astype(np.int16)
            audio_buffer.extend(int16_data)

        try:
            with sd.InputStream(callback=callback,
                              channels=1,
                              samplerate=self.porcupine_sample_rate,
                              dtype=np.float32):
                while self.running:
                    # Check for Esc key
                    if keyboard.is_pressed('esc'):
                        print("\nExiting...")
                        self.running = False
                        break

                    # Wait until we have enough frames
                    while len(audio_buffer) < self.frame_length and self.running:
                        time.sleep(0.01)

                    if not self.running:
                        break

                    # Get frame for Porcupine
                    frame = audio_buffer[:self.frame_length]
                    audio_buffer = audio_buffer[self.frame_length:]

                    # Check for wake word
                    keyword_index = self.porcupine.process(frame)

                    if keyword_index >= 0:
                        print(f"\n✅ Wake word '{self.wake_word}' detected!")

                        # Record audio
                        audio_data = self.record_audio_for_duration(self.recording_duration)

                        # Transcribe
                        text = self.transcribe_audio(audio_data)

                        if text:
                            print(f"\n📝 Transcribed: {text}")
                            pyperclip.copy(text)
                            print("✅ Copied to clipboard!")

                            # Optional: Auto-paste
                            time.sleep(0.2)
                            keyboard.write(text)
                            print("✅ Text pasted!\n")
                        else:
                            print("❌ No text was transcribed\n")

                        print(f"👂 Listening for wake word again...\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        print("Cleaned up resources")

def main():
    # GET YOUR ACCESS KEY FROM: https://console.picovoice.ai/
    ACCESS_KEY = "6B+MM3o9y+cC7CquBjPYHZPRmg0hvra9jdlNJwgN5ZD+nJymGqTgUQ=="  # Replace with your actual key

    # Configuration
    WAKE_WORD = "computer"  # Change to: alexa, computer, jarvis, hey google, etc.
    RECORDING_DURATION = 5  # Seconds to record after wake word

    if ACCESS_KEY == "YOUR_ACCESS_KEY_HERE":
        print("❌ ERROR: Please set your Porcupine access key!")
        print("Get a free key from: https://console.picovoice.ai/")
        print("Then replace 'YOUR_ACCESS_KEY_HERE' in the code with your key.")
        return

    print("Initializing Wake Word Speech-to-Text Transcriber...")
    try:
        transcriber = WakeWordTranscriber(
            access_key=ACCESS_KEY,
            wake_word=WAKE_WORD,
            recording_duration=RECORDING_DURATION
        )
        transcriber.listen_for_wake_word()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

if __name__ == "__main__":
    main()
