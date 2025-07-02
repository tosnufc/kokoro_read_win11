# Kokoro TTS Reader for Windows 11

A text-to-speech application optimized for Windows 11 that reads text from your clipboard using the Kokoro TTS engine.

## Features

- 🎯 **Windows 11 Optimized**: Designed specifically for your system configuration
- 📋 **Clipboard Integration**: Automatically reads text from clipboard
- 🎵 **High-Quality Audio**: Uses Kokoro TTS for natural-sounding speech
- ⚡ **Performance Optimized**: Multi-threaded audio generation and playback
- 🔧 **Error Handling**: Robust error handling and recovery
- 📁 **Audio Export**: Saves generated audio to WAV files

## System Requirements

- ✅ Windows 11 (your system: Windows 10.0.22631)
- ✅ Python 3.8+ (your system: Python 3.11.7)
- ✅ 8GB+ RAM (your system: 16GB)
- ✅ Audio output device

## Quick Setup

### 1. Install Dependencies
Run the setup script to automatically install all required packages:

```bash
python setup_dependencies.py
```

### 2. Alternative Manual Installation
If the setup script fails, install manually:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install core dependencies
pip install numpy>=1.24.0
pip install soundfile>=0.12.1
pip install pyperclip>=1.8.2
pip install pyaudio

# Install Kokoro TTS (try these in order)
pip install kokoro-tts
# OR if the above fails:
pip install git+https://github.com/kokoro-tts/kokoro.git

# Optional for better performance
pip install psutil
```

## Usage

1. **Copy text to clipboard** - Copy any text you want to read aloud
2. **Run the application**:
   ```bash
   python kokoro_readaloud_win11.py
   ```
3. **Listen** - The app will automatically:
   - Read text from clipboard
   - Process and optimize the text
   - Generate high-quality audio
   - Play the audio through your speakers
   - Save audio file to `Output_audio/output.wav`

## Key Optimizations for Your System

### Windows 11 Specific
- **Audio Device Detection**: Automatically finds and uses the best audio output
- **Process Priority**: Sets higher priority for smooth audio playback
- **Resource Management**: Proper cleanup of audio resources

### Performance Optimizations
- **Multi-threading**: Audio generation runs in background thread
- **Chunked Playback**: Streams audio in optimized chunks for your hardware
- **Memory Efficient**: Optimized for your 16GB RAM configuration
- **CPU Optimized**: Takes advantage of your Intel i7-1255U processor

### Audio Quality
- **24kHz Sample Rate**: High-quality audio output
- **Audio Normalization**: Prevents clipping and optimizes volume
- **Intelligent Text Processing**: Better sentence segmentation and pronunciation

## File Structure

```
kokoro-readaloud-win11/
├── kokoro_readaloud_win11.py    # Main TTS application
├── setup_dependencies.py        # Automated dependency installer
├── requirements.txt             # Package dependencies
├── README.md                    # This file
└── Output_audio/               # Generated audio files (created automatically)
    └── output.wav              # Latest generated audio
```

## Troubleshooting

### Common Issues

**"No text found in clipboard"**
- Make sure you've copied text (Ctrl+C) before running the script

**"Audio initialization failed"**
- Check that your audio drivers are up to date
- Ensure no other application is blocking audio access
- Try running as administrator

**"TTS initialization failed"**
- Verify Kokoro TTS is properly installed
- Check internet connection (initial model download)
- Try reinstalling with: `pip uninstall kokoro-tts && pip install kokoro-tts`

**"Import errors"**
- Run the setup script: `python setup_dependencies.py`
- Or install packages manually as shown above

### Performance Tips

1. **Close unnecessary applications** while running TTS for best performance
2. **Use shorter text chunks** (under 1000 words) for faster processing
3. **Keep audio drivers updated** for optimal playback quality

## Advanced Usage

### Voice Selection
The default voice is `af_heart`. You can modify the voice in the code:
```python
audio_segments = self.generate_audio_threaded(processed_text, voice='your_preferred_voice')
```

### Audio Output Location
Audio files are saved to `Output_audio/output.wav`. You can change this in the code or copy files after generation.

## Support

If you encounter issues:
1. Check this README for troubleshooting steps
2. Verify all dependencies are installed correctly
3. Ensure your system meets the requirements
4. Check the console output for specific error messages

---

**Optimized for your Windows 11 system with Intel i7-1255U and 16GB RAM** 🚀 