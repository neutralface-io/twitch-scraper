import os
import m3u8
import requests
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from config import TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, DOWNLOAD_THREADS, WHISPER_MODEL, HUGGINGFACE_API_KEY, USE_GPU
from urllib.parse import urljoin
import concurrent.futures

class TwitchScraper:
    def __init__(self):
        self.client_id = TWITCH_CLIENT_ID
        self.client_secret = TWITCH_CLIENT_SECRET
        self.access_token = None
        self.headers = None
        self.max_workers = DOWNLOAD_THREADS
        self.whisper_model = WHISPER_MODEL
        
        # Initialize models
        try:
            # Initialize pyannote pipeline
            import torch
            from pyannote.audio import Pipeline
            from faster_whisper import WhisperModel
            
            # Set device based on config and availability
            device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
            print(f"\nUsing device: {device} for models")
            
            # Initialize diarization pipeline
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=HUGGINGFACE_API_KEY
            )
            self.diarization_pipeline.to(torch.device(device))
            
            # Initialize faster-whisper
            print(f"\nInitializing Whisper model: {self.whisper_model}")
            
            # Set compute type based on device
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
            
            self.whisper_model = WhisperModel(
                self.whisper_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=4 if device == "cpu" else None
            )
            print(f"Using {compute_type} compute type on {device}")
            
        except Exception as e:
            print(f"\nError initializing models: {e}")
            raise
        
    def authenticate(self) -> None:
        """Authenticate with Twitch API and get access token"""
        auth_url = "https://id.twitch.tv/oauth2/token"
        auth_params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }
        
        response = requests.post(auth_url, params=auth_params)
        response.raise_for_status()
        
        self.access_token = response.json()["access_token"]
        self.headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}"
        }
    
    def get_user_id(self, username: str) -> str:
        """Get Twitch user ID from username"""
        if not self.headers:
            self.authenticate()
            
        url = "https://api.twitch.tv/helix/users"
        params = {"login": username}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        data = response.json()["data"]
        if not data:
            raise ValueError(f"User {username} not found")
            
        return data[0]["id"]
    
    def get_vods(self, username: str) -> List[Dict]:
        """Get all available VODs for a given username"""
        user_id = self.get_user_id(username)
        
        url = "https://api.twitch.tv/helix/videos"
        params = {
            "user_id": user_id,
            "type": "archive",
            "first": 100  # Max items per request
        }
        
        all_vods = []
        
        while True:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            vods = data["data"]
            
            if not vods:
                break
                
            all_vods.extend(vods)
            
            # Check if there are more pages
            pagination = data.get("pagination", {})
            if not pagination.get("cursor"):
                break
                
            params["after"] = pagination["cursor"]
            
        return all_vods

    def get_video_token(self, vod_id: str) -> Dict:
        """Get video access token from Twitch"""
        if not self.headers:
            self.authenticate()
            
        gql_url = "https://gql.twitch.tv/gql"
        headers = {
            "Client-Id": "kimne78kx3ncx6brgo4mv6wki5h1ko",  # Fixed client ID used by web player
            "Client-Version": "1.22.0-rc.0",
            "Client-Session-Id": "ce9e15b8b3c62ba5",
            "X-Device-Id": "68c85a3b280b1652",
        }
        
        query = [{
            "operationName": "PlaybackAccessToken",
            "variables": {
                "isLive": False,
                "login": "",
                "isVod": True,
                "vodID": vod_id,
                "playerType": "embed"
            },
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "0828119ded1c13477966434e15800ff57ddacf13ba1911c129dc2200705b0712"
                }
            }
        }]
        
        response = requests.post(gql_url, json=query, headers=headers)
        response.raise_for_status()
        
        data = response.json()[0]
        return data["data"]["videoPlaybackAccessToken"]

    def get_vod_playlist(self, vod_id: str) -> Optional[str]:
        """Get the m3u8 playlist URL for a VOD"""
        # First get the access token
        access_token = self.get_video_token(vod_id)
        if not access_token:
            return None
            
        # Now get the playlist URL
        params = {
            "sig": access_token["signature"],
            "token": access_token["value"],
            "supported_codecs": "avc1",
            "cdm": "wv",
            "player_version": "1.22.0",
        }
        
        url = f"https://usher.ttvnw.net/vod/{vod_id}.m3u8"
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the m3u8 playlist
        master_playlist = m3u8.loads(response.text)
        
        # Get the lowest quality stream
        if not master_playlist.playlists:
            raise ValueError("No streams found in playlist")
            
        # Print available qualities for debugging
        print("\nAvailable qualities (from highest to lowest):")
        for i, playlist in enumerate(master_playlist.playlists):
            resolution = playlist.stream_info.resolution if playlist.stream_info else 'unknown'
            bandwidth = playlist.stream_info.bandwidth if playlist.stream_info else 'unknown'
            print(f"Quality {i}: {resolution} ({bandwidth/1000:.0f}kbps)")
            
        # Return the last playlist (lowest quality)
        return master_playlist.playlists[-1].uri

    def download_segment(self, segment_info: Tuple[str, str, int]) -> Tuple[int, bytes]:
        """Download a single segment and return its index and data"""
        base_url, segment_uri, index = segment_info
        segment_url = urljoin(base_url, segment_uri)
        
        response = requests.get(segment_url)
        response.raise_for_status()
        
        return index, response.content

    def download_vod(self, vod_id: str, output_path: str, max_workers: int = None) -> None:
        """Download a VOD to the specified path using parallel downloads"""
        # Use instance max_workers if none provided
        max_workers = max_workers or self.max_workers
        
        playlist_url = self.get_vod_playlist(vod_id)
        if not playlist_url:
            raise ValueError(f"Could not get playlist for VOD {vod_id}")
            
        # Get the video segments
        playlist = m3u8.load(playlist_url)
        base_url = playlist_url.rsplit('/', 1)[0] + '/'
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        total_segments = len(playlist.segments)
        print(f"\nDownloading {total_segments} segments using {max_workers} workers...")
        
        # Prepare segment information
        segment_infos = [
            (base_url, segment.uri, i) 
            for i, segment in enumerate(playlist.segments)
        ]
        
        # Download segments in parallel
        downloaded_segments = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Start downloads
            future_to_segment = {
                executor.submit(self.download_segment, segment_info): segment_info[2]
                for segment_info in segment_infos
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=total_segments, desc="Downloading VOD") as pbar:
                for future in concurrent.futures.as_completed(future_to_segment):
                    try:
                        index, data = future.result()
                        downloaded_segments[index] = data
                        pbar.update(1)
                    except Exception as e:
                        print(f"\nError downloading segment {future_to_segment[future]}: {e}")
                        raise
        
        # Write segments to file in order
        print("\nCombining segments...")
        with open(output_path, 'wb') as f:
            for i in range(total_segments):
                if i in downloaded_segments:
                    f.write(downloaded_segments[i])
                else:
                    raise ValueError(f"Missing segment {i}")

    def get_chat_token(self, vod_id: str, offset: int = 0) -> Dict:
        """Get chat access token from Twitch"""
        if not self.headers:
            self.authenticate()
            
        gql_url = "https://gql.twitch.tv/gql"
        headers = {
            "Client-Id": "kimne78kx3ncx6brgo4mv6wki5h1ko",
            "Client-Version": "1.22.0-rc.0",
        }
        
        query = [{
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": {
                "videoID": vod_id,
                "contentOffsetSeconds": offset
            },
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
                }
            }
        }]
        
        response = requests.post(gql_url, json=query, headers=headers)
        response.raise_for_status()
        
        data = response.json()[0].get("data")
        if not data or not data.get("video"):
            raise ValueError(f"Could not get chat data for VOD {vod_id}")
            
        return data["video"]["comments"]

    def parse_duration(self, duration: str) -> int:
        """Convert duration string (e.g. '1h2m3s') to seconds"""
        seconds = 0
        current_num = ""
        
        for char in duration:
            if char.isdigit():
                current_num += char
            elif char == 'h':
                seconds += int(current_num) * 3600
                current_num = ""
            elif char == 'm':
                seconds += int(current_num) * 60
                current_num = ""
            elif char == 's':
                seconds += int(current_num)
                current_num = ""
                
        return seconds

    def download_chat_segment(self, vod_id: str, start: int, end: int) -> List[Tuple[float, str, str]]:
        """Download chat messages for a specific segment of the VOD"""
        comments = set()
        offset = start
        
        while offset <= end:
            cursor = None
            while True:  # Use cursor pagination to get all messages at this offset
                # Get chat messages
                variables = {
                    "videoID": vod_id,
                    "first": 1000
                }
                
                if cursor:
                    variables["cursor"] = cursor
                else:
                    variables["contentOffsetSeconds"] = offset
                
                query = [{
                    "operationName": "VideoCommentsByOffsetOrCursor",
                    "variables": variables,
                    "extensions": {
                        "persistedQuery": {
                            "version": 1,
                            "sha256Hash": "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"
                        }
                    }
                }]
                
                response = requests.post(
                    "https://gql.twitch.tv/gql",
                    json=query,
                    headers={
                        "Client-Id": "kimne78kx3ncx6brgo4mv6wki5h1ko",
                        "Client-Version": "1.22.0-rc.0",
                    }
                )
                response.raise_for_status()
                
                data = response.json()[0].get("data")
                if not data or not data.get("video"):
                    break
                    
                chat_data = data["video"]["comments"]
                if not chat_data or not chat_data.get("edges"):
                    break
                
                # Process comments within our time window
                for edge in chat_data["edges"]:
                    comment = edge["node"]
                    try:
                        timestamp = comment["contentOffsetSeconds"]
                        if start <= timestamp <= end:  # Only include messages in our window
                            message_parts = []
                            for fragment in comment["message"]["fragments"]:
                                if fragment.get("text"):
                                    message_parts.append(fragment["text"])
                                elif fragment.get("emote"):
                                    message_parts.append(fragment["emote"]["token"])
                            
                            message = " ".join(message_parts)
                            comments.add((
                                timestamp,
                                comment["commenter"]["displayName"],
                                message
                            ))
                    except (KeyError, TypeError):
                        continue
                
                # Check if we need to get more messages at this offset
                if not chat_data["pageInfo"].get("hasNextPage"):
                    break
                    
                cursor = chat_data["edges"][-1]["cursor"]
            
            # Move to next offset after getting all messages at current offset
            offset += 10  # s increments
            
        return list(comments)

    def download_chat(self, vod_id: str, output_path: str, duration: str) -> None:
        """Download chat messages for a VOD using parallel processing"""
        print(f"\nDownloading chat for VOD {vod_id}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert duration to seconds
        video_length = self.parse_duration(duration)
        print(f"Video length: {video_length} seconds")
        
        # Split video into segments
        segment_size = 600  # 5 minutes per segment
        segments = [
            (i, min(i + segment_size, video_length))
            for i in range(0, video_length, segment_size)
        ]
        
        all_comments = set()
        
        try:
            with tqdm(total=len(segments), desc="Downloading chat segments") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Start downloads
                    future_to_segment = {
                        executor.submit(self.download_chat_segment, vod_id, start, end): (start, end)
                        for start, end in segments
                    }
                    
                    # Process completed downloads
                    for future in concurrent.futures.as_completed(future_to_segment):
                        try:
                            comments = future.result()
                            all_comments.update(comments)
                            pbar.update(1)
                        except Exception as e:
                            start, end = future_to_segment[future]
                            print(f"\nError downloading segment {start}-{end}: {e}")
                            raise
            
            if not all_comments:
                raise ValueError("No valid chat messages found")
                
            # Sort comments by timestamp
            comments_list = sorted(all_comments, key=lambda x: x[0])
            
            # Write to file
            print(f"\nSaving {len(comments_list)} chat messages...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for timestamp, username, message in comments_list:
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    time_str = f"[{minutes:02d}:{seconds:02d}]"
                    f.write(f'{time_str} {username}: {message}\n')
                    
        except Exception as e:
            print(f"\nError downloading chat: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)  # Clean up partial file
            raise

    def process_audio(self, audio_path: str, transcript_path: str) -> None:
        """Process audio with speaker diarization and transcription"""
        try:
            import numpy as np
            import librosa
            import soundfile as sf
        except ImportError as e:
            raise ImportError(f"Please install required packages: {e}")
            
        print("\nPerforming speaker diarization...")
        try:
            # Perform diarization
            diarization = self.diarization_pipeline(audio_path)
        except Exception as e:
            print(f"\nError during diarization: {e}")
            raise
            
        # Load the full audio file
        print("\nLoading audio file...")
        audio, sr = librosa.load(audio_path, sr=16000)  # 16kHz for Whisper
        
        # Process each speaker segment
        print("\nTranscribing segments...")
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Get segment timestamps and convert to samples
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            
            # Skip very short segments
            if end_sample - start_sample < 1600:  # Skip segments shorter than 100ms
                continue
            
            # Extract the audio segment
            segment_audio = audio[start_sample:end_sample]
            
            # Save segment to temporary file
            temp_path = f"temp_segment_{start_sample}_{end_sample}.wav"
            try:
                sf.write(temp_path, segment_audio, sr)
                
                # Transcribe this segment
                result, _ = self.whisper_model.transcribe(temp_path)
                
                # Get the transcribed text
                text = " ".join([segment.text for segment in result])
                
                # Clean text (remove non-ASCII but keep punctuation)
                text = ''.join(char for char in text if ord(char) < 128)
                
                if text.strip():  # Only add non-empty segments
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker,
                        'text': text.strip()
                    })
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Sort segments by start time to ensure chronological order
        segments.sort(key=lambda x: x['start'])
        
        # Write the combined output
        print(f"\nSaving transcription to: {transcript_path}")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                timestamp = f"[{int(segment['start'])//60:02d}:{int(segment['start'])%60:02d}]"
                f.write(f"{timestamp} {segment['speaker']}: {segment['text']}\n")

    def download_and_transcribe(self, vod_id: str, transcript_path: str, max_workers: int = None) -> None:
        """Download VOD, convert to audio, and process with transcription"""
        try:
            import ffmpeg
        except ImportError as e:
            raise ImportError(f"Please install required packages: {e}")

        max_workers = max_workers or self.max_workers
        
        # Set up output paths
        audio_path = transcript_path.replace('_transcript.txt', '.wav')
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
        # Get playlist
        playlist_url = self.get_vod_playlist(vod_id)
        if not playlist_url:
            raise ValueError(f"Could not get playlist for VOD {vod_id}")
            
        playlist = m3u8.load(playlist_url)
        base_url = playlist_url.rsplit('/', 1)[0] + '/'
        
        total_segments = len(playlist.segments)
        print(f"\nDownloading {total_segments} segments using {max_workers} workers...")
        
        try:
            # Create ffmpeg process
            process = (
                ffmpeg
                .input('pipe:0', f='mpegts')
                .output(
                    audio_path,
                    acodec='pcm_s16le',
                    ac=2,
                    ar='48k'
                )
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            
            # Download and process segments
            downloaded_segments = {}
            with tqdm(total=total_segments, desc="Downloading segments") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_segment = {
                        executor.submit(self.download_segment, (base_url, segment.uri, i)): i
                        for i, segment in enumerate(playlist.segments)
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_segment):
                        index, data = future.result()
                        downloaded_segments[index] = data
                        pbar.update(1)
                        
                        # Write segments in order
                        while len(downloaded_segments) > 0:
                            next_index = min(downloaded_segments.keys())
                            process.stdin.write(downloaded_segments.pop(next_index))
            
            # Finish audio conversion
            process.stdin.close()
            process.wait()
            print(f"\nAudio saved to: {audio_path}")
            
            # Process audio with transcription
            self.process_audio(audio_path, transcript_path)
            
        except Exception as e:
            print(f"\nError processing VOD: {e}")
            # Clean up files on error
            for path in [audio_path, transcript_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise

def main():
    scraper = TwitchScraper()
    
    channel_name = input("Enter Twitch channel name: ")
    try:
        vods = scraper.get_vods(channel_name)
        print(f"\nFound {len(vods)} VODs for channel {channel_name}:")
        for i, vod in enumerate(vods, 1):
            print(f"{i}. Title: {vod['title']}")
            print(f"   ID: {vod['id']}")
            print(f"   Duration: {vod['duration']}")
            print(f"   Created at: {vod['created_at']}")
            print(f"   URL: {vod['url']}\n")
            
        if vods:
            vod_index = int(input("Enter the number of the VOD you want to process (1, 2, etc.): ")) - 1
            if 0 <= vod_index < len(vods):
                vod = vods[vod_index]
                
                # Ask what to download
                print("\nWhat would you like to download?")
                print("1. VOD")
                print("2. Chat")
                print("3. Both")
                choice = input("Enter your choice (1-3): ")
                
                if choice in ('1', '3'):
                    transcript_path = f"downloads/{vod['id']}_transcript.txt"
                    print(f"\nDownloading and transcribing VOD: {vod['title']}")
                    scraper.download_and_transcribe(vod['id'], transcript_path)
                
                if choice in ('2', '3'):
                    chat_path = f"downloads/{vod['id']}_chat.txt"
                    scraper.download_chat(vod['id'], chat_path, vod['duration'])
                    print(f"Chat downloaded successfully to: {chat_path}")
            else:
                print("Invalid VOD number")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 