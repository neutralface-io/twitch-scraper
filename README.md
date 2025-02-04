# Twitch VOD Scraper

A Python tool for downloading, transcribing, and diarizing Twitch VODs and chat logs.

## Features

- List all available VODs from a Twitch channel
- Download and process VODs:
  - Automatically extracts audio
  - Transcribes using OpenAI Whisper
  - Parallel downloading for improved performance
  - Cleaned transcription output
- Download chat logs with timestamps
- Progress tracking for all operations

## Requirements

- Python 3.8+
- FFmpeg installed on your system
- CUDA-capable GPU recommended for faster transcription
- System build tools (for compiling dependencies)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
```bash
choco install ffmpeg
choco install python3
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. Upgrade pip and install build dependencies:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a Twitch application at https://dev.twitch.tv/console
2. Copy your Client ID and Client Secret
3. Update `config.py` with your settings:
   ```python
   # Twitch API credentials
   TWITCH_CLIENT_ID = "your_client_id"
   TWITCH_CLIENT_SECRET = "your_client_secret"
   
   # Download settings
   DOWNLOAD_THREADS = 6  # Number of concurrent download threads
   
   # Whisper settings
   WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
   ```

## Troubleshooting

### Common Installation Issues

1. **Build Dependencies**:
   - If you encounter build errors, ensure you have the necessary system build tools installed
   - On Windows, you might need Visual C++ build tools
   - On Linux, ensure you have python3-dev installed

2. **FFmpeg Issues**:
   - Ensure FFmpeg is properly installed and accessible from the command line
   - Try running `ffmpeg -version` to verify the installation

3. **CUDA/GPU Issues**:
   - Ensure you have CUDA installed if using GPU
   - Check CUDA compatibility with installed PyTorch version

4. **Memory Issues**:
   - For large VODs, ensure you have sufficient RAM
   - Consider using a smaller Whisper model if experiencing memory problems

### Verification

After installation, you can verify the setup by running:
```python
import torch
import whisper

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

## Usage

Run the script:
```bash
python twitch_scraper.py
```

Follow the prompts to:
1. Enter a Twitch channel name
2. Select a VOD from the list
3. Choose to download:
   - VOD (will be transcribed with speaker diarization)
   - Chat logs
   - Both

Downloads will be saved to the `downloads` directory.

## Output Formats

- Audio: 16-bit PCM WAV files (stereo, 48kHz)
- Transcriptions: ASCII-only text files
- Chat: Text files with timestamps in [MM:SS] format

## File Naming

For a VOD with ID `123456789`:
- Audio: `downloads/123456789.wav`
- Transcript: `downloads/123456789_transcript.txt`
- Chat: `downloads/123456789_chat.txt`

## Performance Notes

- Transcription speed depends on your hardware (GPU recommended)
- Downloads are parallelized based on DOWNLOAD_THREADS in config.py
- Audio files are preserved for potential reprocessing

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py
└── twitch_scraper.py
```

## Dependencies

- requests: HTTP requests to Twitch API
- m3u8: Playlist parsing
- tqdm: Progress bars
- ffmpeg-python: Audio processing
- openai-whisper: Speech recognition
- torch: Required for Whisper
- numpy: Required for audio processing

## Troubleshooting

Common issues:
1. "Error initializing diarization pipeline":
   - Check your Hugging Face token
   - Ensure you've accepted the model terms
   - Verify CUDA installation if using GPU
2. Memory issues:
   - Try processing shorter segments
   - Use a smaller Whisper model
   - Switch to CPU if GPU memory is insufficient

## Contributing

Feel free to open issues or submit pull requests.

## License

MIT License 