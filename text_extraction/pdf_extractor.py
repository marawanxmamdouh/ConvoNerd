import os
import shutil

from langchain.document_loaders import PyPDFDirectoryLoader


class PDFTextExtractor:
    def __init__(self, pdf_folder_path="./uploaded_files/pdf", text_folder_path="./uploaded_files/txt"):
        self.pdf_folder_path = pdf_folder_path
        self.text_folder_path = text_folder_path

    def extract_text(self):
        self.clear_text_folder()
        extracted_text = self.extract_text_from_pdf()
        return extracted_text

    def extract_text_from_pdf(self):
        loader = PyPDFDirectoryLoader(self.pdf_folder_path)
        docs = loader.load()
        return docs

    def clear_text_folder(self):
        if os.path.exists(self.text_folder_path):
            shutil.rmtree(self.text_folder_path)
        os.makedirs(self.text_folder_path)
