# Twitch VOD and Chat Scraper

A Python script to download VODs and chat logs from Twitch channels.

## Features

- List all VODs from a Twitch channel
- Download VODs:
  - Automatically selects lowest quality for efficient storage
  - Parallel downloading for better performance
- Download chat logs:
  - Includes timestamps
  - Messages formatted as `[MM:SS] Username: Message`
- Choose to download VOD, chat, or both

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd twitch-scraper
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Twitch API credentials:
   - Go to [Twitch Developer Console](https://dev.twitch.tv/console)
   - Create a new application
   - Get your Client ID and Client Secret
   - Copy them to `config.py`:
     ```python
     TWITCH_CLIENT_ID = "your_client_id_here"
     TWITCH_CLIENT_SECRET = "your_client_secret_here"
     ```

4. Configure download settings in `config.py`:
   ```python
   DOWNLOAD_THREADS = 4  # Number of concurrent download threads
   ```

## Usage

Run the script:
```bash
python twitch_scraper.py
```

The script will:
1. Prompt you for a Twitch channel name
2. Display all available VODs with their:
   - Title
   - VOD ID
   - Duration
   - Creation date
   - URL
3. Let you select which VOD to process
4. Ask whether to download:
   - VOD only
   - Chat only
   - Both VOD and chat
5. Download the selected content to the downloads directory

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py
└── twitch_scraper.py
```

## Dependencies

- requests: For making HTTP requests to Twitch API
- m3u8: For parsing video playlists
- tqdm: For displaying download progress bars

## Contributing

Feel free to open issues or submit pull requests.

## License

MIT License 