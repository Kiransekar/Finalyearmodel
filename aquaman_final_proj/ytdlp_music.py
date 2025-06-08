import os
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import yt_dlp
from fastapi import HTTPException
from functools import lru_cache

from ytdlp_music import VideoResult, MAX_CACHE_SIZE

logger = logging.getLogger("youtube_api")

class YouTubeServiceWithCookies:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "5")))
        self.cookies_file = os.getenv("YOUTUBE_COOKIES_FILE")
        
    def get_ydl_opts(self, download=False):
        """Get YT-DLP options with cookie support"""
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 15,
        }
        
        # Add cookies if available
        if self.cookies_file:
            if os.path.exists(self.cookies_file):
                ydl_opts['cookiefile'] = self.cookies_file
                logger.info(f"Using cookies from: {self.cookies_file}")
            else:
                logger.warning(f"Cookies file specified but not found: {self.cookies_file}")
        
        return ydl_opts
    
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def extract_video_info(self, video_id: str) -> VideoResult:
        """Extract video information and streaming URL from YouTube video ID"""
        start_time = time.time()
        logger.info(f"Extracting info for video: {video_id}")
        
        try:
            # Get YT-DLP options with cookies
            ydl_opts = self.get_ydl_opts()
            
            # Create YT-DLP instance
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                # Get the best audio stream URL
                stream_url = None
                format_info = None
                
                for format in info.get('formats', []):
                    if format.get('acodec') != 'none' and format.get('vcodec') == 'none':
                        stream_url = format.get('url')
                        format_info = format
                        break
                
                # If no audio-only stream is found, get the best available stream
                if not stream_url and info.get('formats'):
                    stream_url = info.get('formats')[-1].get('url')
                    format_info = info.get('formats')[-1]
                
                # Get best thumbnail
                thumbnails = info.get('thumbnails', [])
                thumbnail_url = None
                if thumbnails:
                    # Sort thumbnails by resolution and pick the highest quality
                    thumbnails.sort(
                        key=lambda x: x.get('height', 0) * x.get('width', 0) 
                        if x.get('height') and x.get('width') else 0, 
                        reverse=True
                    )
                    thumbnail_url = thumbnails[0].get('url')
                
                # Extract artist if available
                artist = "Unknown Artist"
                if info.get('artist'):
                    artist = info.get('artist')
                elif info.get('uploader'):
                    artist = info.get('uploader')
                
                duration = time.time() - start_time
                logger.info(f"Video info extracted in {duration:.2f}s: {video_id}")
                
                return VideoResult(
                    id=video_id,
                    title=info.get('title', ''),
                    artists=artist,
                    album=info.get('album'),
                    thumbnail=thumbnail_url,
                    duration=info.get('duration'),
                    stream_url=stream_url,
                    error=""
                )
        except Exception as e:
            logger.error(f"Error extracting video info for {video_id}: {str(e)}")
            return VideoResult(
                id=video_id,
                title="",
                error=f"Error extracting info: {str(e)}"
            )
    
    async def search_videos(self, query: str, limit: int = 3) -> list:
        """Search YouTube for videos matching the query"""
        start_time = time.time()
        logger.info(f"Searching YouTube for: '{query}' (limit: {limit})")
        
        try:
            # YT-DLP options for search with cookies
            ydl_opts = self.get_ydl_opts()
            ydl_opts.update({
                'extract_flat': True,
                'default_search': f'ytsearch{limit}'
            })
            
            # Search for videos
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(query, download=False)
                
                # Extract video IDs from search results
                video_ids = []
                if 'entries' in search_results:
                    for entry in search_results['entries']:
                        if entry.get('id'):
                            video_ids.append(entry.get('id'))
            
            duration = time.time() - start_time
            logger.info(f"Search completed in {duration:.2f}s, found {len(video_ids)} results")
            
            # Use ThreadPoolExecutor to extract video info in parallel
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self.executor, self.extract_video_info, video_id)
                for video_id in video_ids
            ]
            results = await asyncio.gather(*tasks)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching videos: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error searching videos: {str(e)}")
    
    def get_stream_details(self, video_id: str) -> dict:
        """Get detailed streaming information for a video"""
        try:
            # Get YT-DLP options with cookies
            ydl_opts = self.get_ydl_opts()
            
            # Create YT-DLP instance
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                # Get the best audio stream
                for format in info.get('formats', []):
                    if format.get('acodec') != 'none' and format.get('vcodec') == 'none':
                        return {
                            "video_id": video_id,
                            "stream_url": format.get('url'),
                            "format": format.get('format_id', 'unknown'),
                            "quality": f"{format.get('abr', 'unknown')}kbps",
                            "expires": format.get('expires'),
                        }
                
                # If no audio-only stream is found, get the best available stream
                if info.get('formats'):
                    best_format = info.get('formats')[-1]
                    return {
                        "video_id": video_id,
                        "stream_url": best_format.get('url'),
                        "format": best_format.get('format_id', 'unknown'),
                        "quality": f"{best_format.get('format_note', 'unknown')}",
                        "expires": best_format.get('expires'),
                    }
                
                raise HTTPException(status_code=404, detail="No suitable stream found")
                
        except Exception as e:
            logger.error(f"Error getting stream details for {video_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting stream details: {str(e)}")