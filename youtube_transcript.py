import json
import os
import re
import shutil

from loguru import logger as log
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


def extract_video_id(video_input):
    """
    Extracts the video ID from a YouTube video URL or video ID.

    Parameters
    ----------
    video_input : str
        The input string representing a YouTube video URL or video ID.

    Returns
    -------
    str
        The extracted video ID from the input string.

    Examples
    --------
    >>> extract_video_id('www.youtube.com/watch?v=NKfz2EE0fSg')
    ... 'NKfz2EE0fSg'
    """
    # Regular expression to match YouTube video URLs
    url_pattern = re.compile(
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube'
        r'\.com/e/|youtube\.com/user/.*/u/|youtube\.com/s/|youtube\.com/playlist\?list=)([^"&?/\s]{11})'
    )

    # Try to match the input with the URL pattern
    match = url_pattern.match(video_input)

    if match:
        # If it's a URL, return the extracted video ID
        return match.group(1)
    else:
        # If it's not a URL, assume it's a video ID and return it
        return video_input


def save_transcript_as_json(video_id):
    """
    Saves the transcript of a video as a JSON file.

    Parameters
    ----------
    video_id : str
        The ID of the video for which to save the transcript.

    Returns
    -------
    bool
        True if the transcript is successfully saved as a JSON file, False otherwise.

    Raises
    ------
    TranscriptsDisabled
        If the video has no available transcripts or subtitles are disabled.

    Examples
    --------
    >>> save_transcript_as_json('NKfz2EE0fSg')
    ... None
    """
    try:
        # Delete the uploaded_files folder if it exists and create a new one
        shutil.rmtree('uploaded_files') if os.path.isdir('uploaded_files') else None
        os.makedirs('./uploaded_files/json')

        # Get the transcript for the video
        transcripts = YouTubeTranscriptApi.get_transcripts([video_id])

        # Save the transcript to a JSON file
        with open(f'./uploaded_files/json/transcript.json', 'w', encoding='utf-8') as json_file:
            json.dump(transcripts, json_file)

    # Catch exceptions caused by a video having no available transcripts
    except TranscriptsDisabled:
        log.error(f'Subtitles are disabled for the video with ID {video_id}. or this video is not available')
        return False
