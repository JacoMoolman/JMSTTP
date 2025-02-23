import keyboard
import sounddevice as sd
import numpy as np
import whisper
import pyperclip
import wavio
import tempfile
import os
from threading import Thread
import time
from scipy.io import wavfile

class SpeechToTextTranscriber:
    def __init__(self):
        self.model = whisper.load_model("base")
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
                print("Recording... Press Ctrl+Home again to stop.")
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
    transcriber = SpeechToTextTranscriber()
    recording_thread = None
    
    def on_hotkey():
        nonlocal recording_thread
        
        if not transcriber.recording:
            # Start recording
            print("\nStarting new recording...")
            recording_thread = Thread(target=transcriber.record_audio)
            recording_thread.start()
        else:
            # Stop recording and transcribe
            print("\nStopping recording...")
            transcriber.stop_recording()
            recording_thread.join()
            
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

    # Register the hotkey (Ctrl+Home)
    keyboard.add_hotkey('ctrl+home', on_hotkey)
    print("Speech-to-Text is running! Press Ctrl+Home to start/stop recording.")
    print("Press Esc to quit the program.")
    
    
    # Keep the program running
    keyboard.wait('esc')

if __name__ == "__main__":
    main()
