from http.client import InvalidURL

import validators
from langchain.document_loaders.url import UnstructuredURLLoader
from loguru import logger as log


class URLTextExtractor:
    def __init__(self, urls):
        self.urls = urls
        self.extracted_text = []

    def extract_text_from_urls(self):
        valid_urls = self.validate_urls()

        for url in valid_urls:
            extracted_text = self.extract_text_from_url(url)
            self.handle_extracted_text(url, extracted_text)

        self.handle_no_text_extracted()

        return self.extracted_text

    def extract_text_from_url(self, url):
        unstructured_loader = UnstructuredURLLoader([url])
        return unstructured_loader.load()

    def handle_extracted_text(self, url, extracted_text):
        if extracted_text:
            self.extracted_text.extend(extracted_text)
        else:
            log.warning(f'No content was extracted from the URL {url}')

    def handle_no_text_extracted(self):
        if not self.extracted_text:
            log.warning("No content was extracted from the URLs provided")
            raise InvalidURL("No valid content found in the provided URLs")

    def validate_urls(self):
        invalid_urls = []
        for url in self.urls:
            if not validators.url(url):
                invalid_urls.append(url)
                log.warning(f'Invalid URL: {url}')

        if len(invalid_urls) == len(self.urls):
            raise InvalidURL("All URLs are invalid")

        # Return the list of valid URLs
        return list(set(self.urls) - set(invalid_urls))
