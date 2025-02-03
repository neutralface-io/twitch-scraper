import os
import m3u8
import requests
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from config import TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, DOWNLOAD_THREADS
from urllib.parse import urljoin
import concurrent.futures

class TwitchScraper:
    def __init__(self):
        self.client_id = TWITCH_CLIENT_ID
        self.client_secret = TWITCH_CLIENT_SECRET
        self.access_token = None
        self.headers = None
        self.max_workers = DOWNLOAD_THREADS
        
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

    def get_chat_token(self, vod_id: str, offset: int = 0, cursor: str = None) -> Dict:
        """Get chat access token from Twitch"""
        if not self.headers:
            self.authenticate()
            
        gql_url = "https://gql.twitch.tv/gql"
        headers = {
            "Client-Id": "kimne78kx3ncx6brgo4mv6wki5h1ko",
            "Client-Version": "1.22.0-rc.0",
        }
        
        variables = {
            "videoID": vod_id,
            "first": 100,  # Number of messages per request
        }
        
        # Use cursor if provided, otherwise use offset
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
        
        response = requests.post(gql_url, json=query, headers=headers)
        response.raise_for_status()
        
        data = response.json()[0].get("data")
        if not data or not data.get("video"):
            raise ValueError(f"Could not get chat data for VOD {vod_id}")
            
        return data["video"]["comments"]

    def download_chat(self, vod_id: str, output_path: str) -> None:
        """Download chat messages for a VOD"""
        print(f"\nDownloading chat for VOD {vod_id}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        comments = []
        offset = 0
        cursor = None
        
        try:
            with tqdm(desc="Downloading chat messages") as pbar:
                while True:
                    # Get chat messages starting from offset
                    chat_data = self.get_chat_token(vod_id, offset, cursor)
                    
                    if not chat_data or not chat_data.get("edges"):
                        break
                    
                    # Process comments
                    for edge in chat_data["edges"]:
                        comment = edge["node"]
                        try:
                            # Handle message fragments (emotes, text, etc.)
                            message_parts = []
                            for fragment in comment["message"]["fragments"]:
                                if fragment.get("text"):
                                    message_parts.append(fragment["text"])
                                elif fragment.get("emote"):
                                    message_parts.append(fragment["emote"]["token"])
                            
                            message = " ".join(message_parts)
                            
                            comments.append({
                                "timestamp": comment["contentOffsetSeconds"],
                                "username": comment["commenter"]["displayName"],
                                "message": message,
                            })
                            pbar.update(1)
                            
                            # Update offset to latest message timestamp
                            offset = max(offset, comment["contentOffsetSeconds"])
                        except (KeyError, TypeError):
                            continue
                    
                    # Check for next page
                    if chat_data["pageInfo"].get("hasNextPage"):
                        cursor = chat_data["edges"][-1]["cursor"]
                    else:
                        # Reset cursor and move offset forward
                        cursor = None
                        offset += 1
                        
                    # Break if we've collected a significant number of messages
                    if len(comments) > 100000:
                        break
            
            if not comments:
                raise ValueError("No valid chat messages found")
                
            # Sort comments by timestamp
            comments.sort(key=lambda x: x["timestamp"])
            
            # Write to file
            print(f"\nSaving {len(comments)} chat messages...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for comment in comments:
                    minutes = int(comment["timestamp"] // 60)
                    seconds = int(comment["timestamp"] % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    f.write(f'{timestamp} {comment["username"]}: {comment["message"]}\n')
                    
        except Exception as e:
            print(f"\nError downloading chat: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)  # Clean up partial file
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
                    output_path = f"downloads/{vod['id']}.mp4"
                    print(f"\nDownloading VOD: {vod['title']}")
                    scraper.download_vod(vod['id'], output_path)
                    print(f"VOD downloaded successfully to: {output_path}")
                
                if choice in ('2', '3'):
                    chat_path = f"downloads/{vod['id']}_chat.txt"
                    scraper.download_chat(vod['id'], chat_path)
                    print(f"Chat downloaded successfully to: {chat_path}")
            else:
                print("Invalid VOD number")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 