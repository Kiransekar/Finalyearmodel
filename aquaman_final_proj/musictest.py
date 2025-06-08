import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from ytdlp_music import app, YouTubeService, VideoResult, get_youtube_service

# Create a test client
client = TestClient(app)

# Sample test data
SAMPLE_VIDEO_ID = "dQw4w9WgXcQ"
SAMPLE_SEARCH_QUERY = "rick astley never gonna give you up"
SAMPLE_VIDEO_RESULT = VideoResult(
    id=SAMPLE_VIDEO_ID,
    title="Rick Astley - Never Gonna Give You Up",
    artists="Rick Astley",
    album="Whenever You Need Somebody",
    thumbnail="https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
    duration=213,
    stream_url="https://example.com/stream",
    error=""
)
SAMPLE_STREAM_DETAILS = {
    "video_id": SAMPLE_VIDEO_ID,
    "stream_url": "https://example.com/stream",
    "format": "251",
    "quality": "160kbps",
    "expires": "20250516"
}

# Mock the YouTube service for testing
@pytest.fixture
def mock_youtube_service():
    with patch("ytdlp_music.get_youtube_service") as mock_get_service:
        service_mock = MagicMock(spec=YouTubeService)
        
        # Mock the extract_video_info method
        service_mock.extract_video_info.return_value = SAMPLE_VIDEO_RESULT
        
        # Mock the search_videos method
        async def mock_search(*args, **kwargs):
            return [SAMPLE_VIDEO_RESULT]
        service_mock.search_videos = mock_search
        
        # Mock the get_stream_details method
        service_mock.get_stream_details.return_value = SAMPLE_STREAM_DETAILS
        
        mock_get_service.return_value = service_mock
        yield service_mock

# Test the health check endpoint
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# Test the video info endpoint with mocked service
def test_get_video_info(mock_youtube_service):
    response = client.get(f"/video/{SAMPLE_VIDEO_ID}")
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == SAMPLE_VIDEO_ID
    assert result["title"] == "Rick Astley - Never Gonna Give You Up"
    mock_youtube_service.extract_video_info.assert_called_once_with(SAMPLE_VIDEO_ID)

# Test the streaming URL endpoint with mocked service
def test_get_streaming_url(mock_youtube_service):
    response = client.get(f"/stream/{SAMPLE_VIDEO_ID}")
    assert response.status_code == 200
    result = response.json()
    assert result["video_id"] == SAMPLE_VIDEO_ID
    assert result["stream_url"] == "https://example.com/stream"
    mock_youtube_service.get_stream_details.assert_called_once_with(SAMPLE_VIDEO_ID)

# Test the search endpoint with mocked service
@pytest.mark.asyncio
async def test_search_videos(mock_youtube_service):
    response = client.post("/search", json={"query": SAMPLE_SEARCH_QUERY, "limit": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["query"] == SAMPLE_SEARCH_QUERY
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == SAMPLE_VIDEO_ID

# Test error handling for video info endpoint
def test_get_video_info_error(mock_youtube_service):
    # Set up the mock to return an error
    error_result = VideoResult(
        id=SAMPLE_VIDEO_ID,
        title="",
        error="Video not found"
    )
    mock_youtube_service.extract_video_info.return_value = error_result
    
    response = client.get(f"/video/{SAMPLE_VIDEO_ID}")
    assert response.status_code == 404
    assert "Video not found" in response.json()["detail"]

# Test error handling for streaming URL endpoint
def test_get_streaming_url_error(mock_youtube_service):
    # Set up the mock to raise an HTTPException
    from fastapi import HTTPException
    mock_youtube_service.get_stream_details.side_effect = HTTPException(
        status_code=404, detail="No suitable stream found"
    )
    
    response = client.get(f"/stream/{SAMPLE_VIDEO_ID}")
    assert response.status_code == 404
    assert "No suitable stream found" in response.json()["detail"]

# Integration tests with real YouTube API (these will be skipped by default)
@pytest.mark.skip(reason="Integration test that requires internet connection")
def test_integration_video_info():
    response = client.get(f"/video/{SAMPLE_VIDEO_ID}")
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == SAMPLE_VIDEO_ID
    assert "Rick Astley" in result["title"]

@pytest.mark.skip(reason="Integration test that requires internet connection")
def test_integration_search():
    response = client.post("/search", json={"query": SAMPLE_SEARCH_QUERY, "limit": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["query"] == SAMPLE_SEARCH_QUERY
    assert len(result["results"]) > 0

# Test service class methods directly
class TestYouTubeService:
    @pytest.fixture
    def youtube_service(self):
        return YouTubeService()
    
    @pytest.mark.asyncio
    @patch("yt_dlp.YoutubeDL")
    async def test_search_videos(self, mock_ydl, youtube_service):
        # Mock YoutubeDL extract_info
        mock_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "entries": [{"id": SAMPLE_VIDEO_ID}]
        }
        
        # Mock extract_video_info method
        youtube_service.extract_video_info = MagicMock(return_value=SAMPLE_VIDEO_RESULT)
        
        results = await youtube_service.search_videos(SAMPLE_SEARCH_QUERY, 1)
        assert len(results) == 1
        assert results[0].id == SAMPLE_VIDEO_ID
        
        # Verify the method was called with the right parameters
        mock_instance.extract_info.assert_called_once_with(SAMPLE_SEARCH_QUERY, download=False)
    
    @patch("yt_dlp.YoutubeDL")
    def test_extract_video_info(self, mock_ydl, youtube_service):
        # Mock YoutubeDL extract_info
        mock_instance = MagicMock()
        mock_ydl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            "id": SAMPLE_VIDEO_ID,
            "title": "Rick Astley - Never Gonna Give You Up",
            "artist": "Rick Astley",
            "album": "Whenever You Need Somebody",
            "duration": 213,
            "thumbnails": [{"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg", "width": 1280, "height": 720}],
            "formats": [
                {"url": "https://example.com/stream", "acodec": "mp4a.40.2", "vcodec": "none"}
            ]
        }
        
        result = youtube_service.extract_video_info(SAMPLE_VIDEO_ID)
        assert result.id == SAMPLE_VIDEO_ID
        assert result.title == "Rick Astley - Never Gonna Give You Up"
        assert result.artists == "Rick Astley"
        assert result.stream_url == "https://example.com/stream"
        
        # Verify the method was called with the right parameters
        mock_instance.extract_info.assert_called_once_with(
            f"https://www.youtube.com/watch?v={SAMPLE_VIDEO_ID}", 
            download=False
        )

# Run tests with pytest
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])