# Importing the necessary libraries
import json
import os
import re
import shutil

from loguru import logger as log
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from text_extraction.Json_extractor import JsonTextExtractor


class YouTubeTextExtractor:
    """Class for extracting text from a YouTube video via its URL or ID."""

    def __init__(self):
        """Initialize the extractor, specifying the path to save the resulting transcript."""
        self.transcript_file_path = "./uploaded_files/json/transcript.json"

    def extract_text(self, video_url):
        """
        Extract text from a given YouTube video.

        Parameters
        ----------
        video_url : str
            The URL or ID of the YouTube video from which to extract text.

        Returns
        -------
        str
            The extracted text.
        """
        video_id = self.extract_video_id(video_url)
        self.save_transcript_as_json(video_id)

        self.convert_transcript_to_txt()
        return self.load_text_from_file()

    def extract_video_id(self, video_input):
        """
        Extract the ID from a given YouTube input.

        Parameters
        ----------
        video_input : str
            The input, which can be a YouTube URL or an ID of a video.

        Returns
        -------
        str
            The ID of the YouTube video.
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

    def save_transcript_as_json(self, video_id):
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
            raise TranscriptsDisabled(f'Subtitles are disabled for the video with ID {video_id}. or this video is not '
                                      f'available')

    def convert_transcript_to_txt(self):
        """Convert a JSON transcript to text."""
        if os.path.exists(self.transcript_file_path):
            json_text_extractor = JsonTextExtractor()
            json_text_extractor.convert_transcript_to_txt()

    def load_text_from_file(self):
        """
        Load text from a pre-determined text file.

        Returns
        -------
        str
            The contents of the text file as a string.
        """
        with open("./uploaded_files/txt/transcript.txt", "r", encoding="utf-8") as txt_file:
            return txt_file.read()
