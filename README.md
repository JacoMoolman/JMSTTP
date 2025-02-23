# Speech-to-Text Hotkey

A simple Windows application that converts speech to text using OpenAI's Whisper model. Press a hotkey to start recording, speak your message, and press the hotkey again to have your speech transcribed and automatically pasted where your cursor is.

## Features
- Quick speech-to-text conversion using Ctrl+Home hotkey
- Uses OpenAI's Whisper model for accurate transcription
- Automatically pastes transcribed text
- Saves latest recording as WAV file for verification

## Requirements
- Python 3.8 or higher
- Required Python packages:
  - keyboard
  - sounddevice
  - numpy
  - openai-whisper
  - pyperclip
  - wavio
  - scipy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install keyboard sounddevice numpy openai-whisper pyperclip wavio scipy
```

## Usage
1. Run `run_speech_to_text.bat` or create a shortcut to it
2. Press `Ctrl+Home` to start recording
3. Speak your message
4. Press `Ctrl+Home` again to stop recording and paste the transcribed text
5. Press `Esc` to quit the program
