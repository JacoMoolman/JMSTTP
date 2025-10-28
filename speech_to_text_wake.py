import subprocess
import sys
import importlib.util
import os
import json

# ============================================================================
# CONFIGURATION - Adjust these settings to fit your environment
# ============================================================================

# Porcupine Wake Word Settings
# GET YOUR ACCESS KEY FROM: https://console.picovoice.ai/
ACCESS_KEY = "6B+MM3o9y+cC7CquBjPYHZPRmg0hvra9jdlNJwgN5ZD+nJymGqTgUQ=="

# Wake word to detect
# Built-in keywords: alexa, americano, blueberry, bumblebee, computer,
# grapefruit, grasshopper, hey google, hey siri, jarvis, ok google,
# picovoice, porcupine, terminator
WAKE_WORD = "jarvis"

# Voice Activity Detection (VAD) Settings
# Energy threshold for speech detection (lower = more sensitive)
# Typical range: 0.001-0.01
# - If it stops too early during pauses: LOWER this value (e.g., 0.003)
# - If it doesn't detect silence: RAISE this value (e.g., 0.008)
# - Run with debug output to see your actual energy levels
ENERGY_THRESHOLD = 0.005

# Duration of silence (in seconds) before auto-stopping recording
# - Increase if you have natural pauses in speech (e.g., 2.0 or 2.5)
# - Decrease for faster response (e.g., 1.0)
SILENCE_DURATION = 4

# Maximum recording duration in seconds (safety limit)
MAX_RECORDING_DURATION = 60

# Sound effect to play when wake word is detected
# Set to None to disable sound effect
ACTIVATION_SOUND = "click.wav"

# Command configuration file
COMMANDS_CONFIG = "commands.json"

# Command prefixes - if transcription starts with these, treat as command
# Otherwise, treat as dictation even if keywords are present
COMMAND_PREFIXES = ["command", "execute", "run", "do"]

# ============================================================================


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

def load_commands_config(config_file):
    """Load command mappings from JSON config file."""
    try:
        if not os.path.exists(config_file):
            print(f"Warning: Commands config file '{config_file}' not found")
            print("Creating default config file...")
            default_config = {
                "commands": [
                    {
                        "keywords": ["open notepad", "open note pad", "notepad", "note pad"],
                        "command": "notepad.exe",
                        "args": [],
                        "working_dir": None,
                        "description": "Opens Notepad"
                    }
                ]
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

        with open(config_file, 'r') as f:
            config = json.load(f)

        if 'commands' not in config:
            print(f"Error: Invalid config format in '{config_file}'")
            return {"commands": []}

        print(f"Loaded {len(config['commands'])} command(s) from config")
        for cmd in config['commands']:
            keywords = cmd.get('keywords', [])
            if not keywords:
                keywords = [cmd.get('keyword', 'Unknown')]
            keywords_str = "', '".join(keywords)
            print(f"  - ['{keywords_str}']: {cmd.get('description', 'No description')}")

        return config
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in '{config_file}': {e}")
        return {"commands": []}
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {"commands": []}

def execute_command(command_config):
    """Execute a command from the config."""
    try:
        command_type = command_config.get('type', 'process')  # Default to 'process' for backward compatibility
        description = command_config.get('description', 'Unknown command')

        print(f"🚀 Executing: {description}")

        if command_type == 'keyboard':
            # Handle keyboard shortcut
            shortcut = command_config.get('shortcut', '')
            if not shortcut:
                print(f"❌ Error: No shortcut specified for keyboard command")
                return False

            print(f"   Keyboard shortcut: {shortcut}")

            # Import keyboard module (already imported at top)
            import keyboard

            # Press the keyboard shortcut
            keyboard.send(shortcut)
            print(f"✅ Keyboard shortcut executed successfully")
            return True

        elif command_type == 'process':
            # Handle process execution (original functionality)
            command = command_config.get('command')
            args = command_config.get('args', [])
            working_dir = command_config.get('working_dir')

            print(f"   Command: {command} {' '.join(args)}")

            # Build full command
            full_command = [command] + args

            # Execute the command
            if working_dir and os.path.exists(working_dir):
                result = subprocess.Popen(full_command, cwd=working_dir)
            else:
                result = subprocess.Popen(full_command)

            print(f"✅ Command executed successfully (PID: {result.pid})")
            return True

        else:
            print(f"❌ Error: Unknown command type: {command_type}")
            return False

    except FileNotFoundError:
        command = command_config.get('command', 'Unknown')
        print(f"❌ Error: Command not found: {command}")
        return False
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return False

def match_keyword(text, commands_config, command_prefixes):
    """Match transcribed text against configured keywords.

    Args:
        text: Transcribed text to check
        commands_config: Dictionary containing command configurations
        command_prefixes: List of prefixes that indicate a command (e.g., ["command", "execute"])

    Returns:
        Tuple of (matched_command, cleaned_text) or (None, original_text)
    """
    import re

    text_lower = text.lower().strip()

    # Check if text starts with a command prefix
    has_prefix = False
    cleaned_text = text_lower

    for prefix in command_prefixes:
        prefix_lower = prefix.lower()
        # Check if text starts with the prefix (with word boundary)
        if text_lower.startswith(prefix_lower + " ") or text_lower == prefix_lower:
            has_prefix = True
            # Remove the prefix from the text for matching
            cleaned_text = text_lower[len(prefix_lower):].strip()
            break

    # If no command prefix found, return None (treat as dictation)
    if not has_prefix:
        return None, text

    # Now match keywords in the cleaned text (without prefix)
    # First pass: Try exact match
    for cmd in commands_config.get('commands', []):
        keywords = cmd.get('keywords', [])
        if not keywords:
            single_keyword = cmd.get('keyword', '')
            if single_keyword:
                keywords = [single_keyword]

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and cleaned_text == keyword_lower:
                return cmd, cleaned_text

    # Second pass: Try word boundary match (whole word match)
    for cmd in commands_config.get('commands', []):
        keywords = cmd.get('keywords', [])
        if not keywords:
            single_keyword = cmd.get('keyword', '')
            if single_keyword:
                keywords = [single_keyword]

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower:
                # Use word boundaries to match whole words/phrases
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, cleaned_text):
                    return cmd, cleaned_text

    # Third pass: Substring match (original behavior as fallback)
    for cmd in commands_config.get('commands', []):
        keywords = cmd.get('keywords', [])
        if not keywords:
            single_keyword = cmd.get('keyword', '')
            if single_keyword:
                keywords = [single_keyword]

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if keyword_lower and keyword_lower in cleaned_text:
                return cmd, cleaned_text

    return None, text

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
    def __init__(self, access_key, wake_word, energy_threshold=0.02,
                 silence_duration=1.5, max_recording_duration=30, activation_sound=None,
                 commands_config=None, command_prefixes=None):
        """
        Initialize with Porcupine access key.
        Get free key from: https://console.picovoice.ai/

        Built-in keywords: alexa, americano, blueberry, bumblebee, computer,
        grapefruit, grasshopper, hey google, hey siri, jarvis, ok google,
        picovoice, porcupine, terminator

        Args:
            access_key: Porcupine access key
            wake_word: Wake word to detect
            energy_threshold: Energy level threshold for speech detection (0.01-0.1)
            silence_duration: Seconds of silence before stopping recording
            max_recording_duration: Maximum recording time in seconds
            activation_sound: Path to WAV file to play when wake word is detected
            commands_config: Dictionary containing command mappings
            command_prefixes: List of prefixes that indicate a command
        """
        print("Loading Whisper model...")
        self.model = whisper.load_model("medium")
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

        print("Initializing Voice Activity Detection (Energy-based)...")

        self.wake_word = wake_word
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.max_recording_duration = max_recording_duration
        self.activation_sound = activation_sound
        self.commands_config = commands_config if commands_config else {"commands": []}
        self.command_prefixes = command_prefixes if command_prefixes else ["command", "execute", "run"]
        self.sample_rate = 16000
        self.porcupine_sample_rate = self.porcupine.sample_rate
        self.frame_length = self.porcupine.frame_length
        self.wav_filename = "latest_recording.wav"
        self.running = True

        # VAD frame size (100ms chunks for analysis)
        self.vad_frame_duration_ms = 100
        self.vad_frame_size = int(self.sample_rate * self.vad_frame_duration_ms / 1000)

    def play_activation_sound(self):
        """Play the activation sound effect if configured."""
        if self.activation_sound and os.path.exists(self.activation_sound):
            try:
                # Load and play the WAV file
                rate, data = wavfile.read(self.activation_sound)
                sd.play(data, rate)
                sd.wait()  # Wait until sound finishes playing
            except Exception as e:
                print(f"Warning: Could not play sound '{self.activation_sound}': {e}")
        elif self.activation_sound:
            print(f"Warning: Sound file '{self.activation_sound}' not found")

    def calculate_energy(self, audio_frame):
        """Calculate RMS energy of audio frame."""
        return np.sqrt(np.mean(audio_frame ** 2))

    def is_speech(self, audio_frame):
        """Simple energy-based voice activity detection."""
        energy = self.calculate_energy(audio_frame)
        return energy > self.energy_threshold

    def record_audio_with_vad(self):
        """Record audio and automatically stop when speech ends."""
        print(f"🎤 Recording... (will auto-stop after {self.silence_duration}s of silence)")
        audio_data = []
        is_speech_detected = False
        silence_start = None
        recording_start = time.time()
        last_check_index = 0

        def callback(indata, frames, time_info, status):
            if status:
                print(f'Status: {status}')
            audio_data.extend(indata[:, 0])

        stream = sd.InputStream(callback=callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          dtype=np.float32)

        stream.start()

        try:
            while True:
                current_time = time.time()
                elapsed = current_time - recording_start

                # Check if we've exceeded max recording duration
                if elapsed > self.max_recording_duration:
                    print(f"\n⏱️  Maximum recording duration ({self.max_recording_duration}s) reached")
                    break

                # Wait for enough new audio data to analyze
                # Sleep for the duration of one VAD frame
                time.sleep(self.vad_frame_duration_ms / 1000.0)

                # Check if we have enough new audio data
                current_length = len(audio_data)
                if current_length < last_check_index + self.vad_frame_size:
                    continue

                # Get the most recent complete frame for VAD
                recent_audio = np.array(audio_data[last_check_index:last_check_index + self.vad_frame_size], dtype=np.float32)
                last_check_index += self.vad_frame_size

                # Check if current frame contains speech using energy-based VAD
                energy = self.calculate_energy(recent_audio)
                speech_detected = self.is_speech(recent_audio)

                # Debug output every 10 frames (~1 second)
                if last_check_index % (self.vad_frame_size * 10) == self.vad_frame_size:
                    print(f"🔊 Energy: {energy:.4f} (threshold: {self.energy_threshold:.4f}) - {'SPEECH' if speech_detected else 'SILENCE'}")

                if speech_detected:
                    if not is_speech_detected:
                        print(f"🗣️  Speech detected! (energy: {energy:.4f})")
                        is_speech_detected = True
                    silence_start = None  # Reset silence timer
                else:
                    # Only start counting silence after we've detected speech
                    if is_speech_detected:
                        if silence_start is None:
                            silence_start = current_time
                            print(f"🤫 Silence detected (energy: {energy:.4f}), waiting {self.silence_duration}s...")
                        elif current_time - silence_start >= self.silence_duration:
                            print(f"\n🤫 {self.silence_duration}s of silence detected, stopping recording")
                            break

        finally:
            stream.stop()
            stream.close()

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
        print(f"Will auto-stop recording after {self.silence_duration}s of silence")
        print(f"Maximum recording duration: {self.max_recording_duration}s")
        print("Press Esc to quit")
        print(f"{'='*60}\n")

        audio_buffer = []
        frame_count = 0

        def callback(indata, frames, time_info, status):
            if status:
                print(f'Status: {status}')
            # Convert float32 to int16 for Porcupine
            int16_data = (indata[:, 0] * 32767).astype(np.int16)
            audio_buffer.extend(int16_data)

        try:
            print("DEBUG: About to open audio stream...")
            with sd.InputStream(callback=callback,
                              channels=1,
                              samplerate=self.porcupine_sample_rate,
                              dtype=np.float32):
                print("DEBUG: Audio stream opened!")
                print(f"DEBUG: Sample rate: {self.porcupine_sample_rate}, Frame length: {self.frame_length}")
                print("DEBUG: Entering main loop...\n")

                while self.running:
                    # Check for Esc key with try/except to catch any issues
                    try:
                        if keyboard.is_pressed('esc'):
                            print("\nDEBUG: Escape key detected")
                            print("\nExiting...")
                            self.running = False
                            break
                    except Exception as e:
                        print(f"DEBUG: Error checking keyboard: {e}")
                        # Continue anyway - don't exit

                    # Wait until we have enough frames
                    while len(audio_buffer) < self.frame_length and self.running:
                        time.sleep(0.01)

                    if not self.running:
                        break

                    # Get frame for Porcupine
                    frame = audio_buffer[:self.frame_length]
                    audio_buffer = audio_buffer[self.frame_length:]

                    # Debug: Show we're processing frames
                    frame_count += 1
                    if frame_count == 1:
                        print(f"DEBUG: Processing first frame!")
                    if frame_count % 100 == 0:
                        print(f"DEBUG: Processed {frame_count} frames - still listening...")

                    # Check for wake word
                    keyword_index = self.porcupine.process(frame)

                    if keyword_index >= 0:
                        print(f"\n✅ Wake word '{self.wake_word}' detected!")

                        # Play activation sound
                        self.play_activation_sound()

                        # Record audio with VAD
                        audio_data = self.record_audio_with_vad()

                        # Transcribe
                        text = self.transcribe_audio(audio_data)

                        if text:
                            print(f"\n📝 Transcribed: {text}")

                            # Check if transcribed text matches any command keyword
                            matched_command, cleaned_text = match_keyword(text, self.commands_config, self.command_prefixes)

                            if matched_command:
                                # Get the first keyword for display
                                keywords = matched_command.get('keywords', [])
                                if not keywords:
                                    keywords = [matched_command.get('keyword', 'Unknown')]
                                print(f"✅ Keyword matched: '{keywords[0]}'")
                                execute_command(matched_command)
                            else:
                                # No keyword match - use original dictation functionality
                                print(f"ℹ️  No command prefix detected - using dictation mode")
                                pyperclip.copy(text)
                                print("✅ Copied to clipboard!")

                                # Auto-paste
                                time.sleep(0.2)
                                keyboard.write(text)
                                print("✅ Text pasted!\n")
                        else:
                            print("❌ No text was transcribed\n")

                        print(f"\n👂 Listening for wake word again...\n")

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
    if ACCESS_KEY == "YOUR_ACCESS_KEY_HERE":
        print("❌ ERROR: Please set your Porcupine access key!")
        print("Get a free key from: https://console.picovoice.ai/")
        print("Then update the ACCESS_KEY variable at the top of this script.")
        return

    print("Initializing Wake Word Command Executor...")
    print(f"Configuration:")
    print(f"  Wake Word: {WAKE_WORD}")
    print(f"  Energy Threshold: {ENERGY_THRESHOLD}")
    print(f"  Silence Duration: {SILENCE_DURATION}s")
    print(f"  Max Recording Duration: {MAX_RECORDING_DURATION}s")
    print()

    # Load commands configuration
    print(f"Loading commands from '{COMMANDS_CONFIG}'...")
    commands_config = load_commands_config(COMMANDS_CONFIG)
    print()

    try:
        transcriber = WakeWordTranscriber(
            access_key=ACCESS_KEY,
            wake_word=WAKE_WORD,
            energy_threshold=ENERGY_THRESHOLD,
            silence_duration=SILENCE_DURATION,
            max_recording_duration=MAX_RECORDING_DURATION,
            activation_sound=ACTIVATION_SOUND,
            commands_config=commands_config,
            command_prefixes=COMMAND_PREFIXES
        )
        transcriber.listen_for_wake_word()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

if __name__ == "__main__":
    main()
