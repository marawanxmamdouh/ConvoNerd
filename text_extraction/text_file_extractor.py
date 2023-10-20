from langchain.document_loaders import TextLoader
from loguru import logger as log


class TextFileExtractor:
    def __init__(self, txt_folder_path="./uploaded_files/txt", md_folder_path="./uploaded_files/md"):
        self.txt_folder_path = txt_folder_path
        self.md_folder_path = md_folder_path

    def extract_text(self, file_name):
        if file_name:
            file_path = self.get_file_path(file_name)
            loader = TextLoader(file_path)
            extracted_text = loader.load()
            return extracted_text
        else:
            log.error("No files found.")
            raise FileNotFoundError("No files found.")

    def get_file_path(self, file_name):
        if file_name.endswith(".txt"):
            return self.txt_folder_path + "/" + file_name
        elif file_name.endswith(".md"):
            return self.md_folder_path + "/" + file_name
        else:
            log.error("File type not supported.")
            raise TypeError("File type not supported.")
