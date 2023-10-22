# Importing the necessary libraries
from http.client import InvalidURL

import validators
from langchain.document_loaders.url import UnstructuredURLLoader
from loguru import logger as log


class URLTextExtractor:
    """Class for extracting text from one or multiple URLs."""

    def __init__(self, urls):
        """
        Initialize the extractor, specifying the urls to extract from.

        Parameters
        ----------
        urls: list
            A list of URLs from which the text should be extracted.
        """
        self.urls = urls
        self.extracted_text = []

    def extract_text_from_urls(self):
        """
        Extracts text from all valid URLs provided and handles exceptions.

        Returns
        -------
        extracted_text: list
            A list of all texts extracted from the URLs.

        Raises
        ------
        InvalidURL
            If no valid content can be found in the provided URLs.
        """
        valid_urls = self.validate_urls()

        for url in valid_urls:
            extracted_text = self.extract_text_from_url(url)
            self.handle_extracted_text(url, extracted_text)

        self.handle_no_text_extracted()

        return self.extracted_text

    def extract_text_from_url(self, url):
        """
        Uses an UnstructuredURLLoader to extract text from a single URL.

        Parameters
        ----------
        url: str
            The URL from which to extract the text.

        Returns
        -------
        extracted_text : list
            The text extracted from the URL.
        """
        unstructured_loader = UnstructuredURLLoader([url])
        return unstructured_loader.load()

    def handle_extracted_text(self, url, extracted_text):
        """
        Appends extracted a text to the extractor's content and logs a warning if no text was extracted.

        Parameters
        ----------
        url: str
            The URL from which the text was extracted.
        extracted_text: str
            The extracted text.
        """
        if extracted_text:
            self.extracted_text.extend(extracted_text)
        else:
            log.warning(f'No content was extracted from the URL {url}')

    def handle_no_text_extracted(self):
        """
        Logs a warning and raises an exception if no text was extracted from the URLs.

        Raises
        ------
        InvalidURL
            If no valid content can be found in the provided URLs.
        """
        if not self.extracted_text:
            log.warning("No content was extracted from the URLs provided")
            raise InvalidURL("No valid content found in the provided URLs")

    def validate_urls(self):
        """
        Validates the URLs provided, logs a warning and removes invalid URLs.

        Raises
        ------
        InvalidURL
            If all provided URLs are invalid.

        Returns
        -------
        valid_urls : list
            A list of all valid URLs.
        """
        invalid_urls = []
        for url in self.urls:
            if not validators.url(url):
                invalid_urls.append(url)
                log.warning(f'Invalid URL: {url}')

        if len(invalid_urls) == len(self.urls):
            raise InvalidURL("All URLs are invalid")

        # Return the list of valid URLs
        return list(set(self.urls) - set(invalid_urls))
