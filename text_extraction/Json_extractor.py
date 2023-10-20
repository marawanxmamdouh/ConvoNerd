import json
import os
import shutil

from loguru import logger as log


class JsonTextExtractor:
    """
    A class for extracting text from JSON files and converting the extracted text into single merged text.

    Attributes
    ----------
    directory_path : str
        The directory path where the JSON files are located.
    merged_text : list
        A list to store the merged text from the JSON files.

    Methods
    -------
    __init__(self)
        Initializes the JsonTextExtractor object.
    extract_text_recursive(self, data)
        Recursively extracts text from JSON data.
    load_and_merge_json_files(self)
        Loads and merges the JSON files.
    get_merged_text(self)
        Returns the merged text.
    convert_transcript_to_txt(self)
        Converts the transcript to a txt file and saves it in the txt folder.

    Notes
    -----
    This assumes that the JSON files are located in the 'uploaded_files/json' folder relative to this script.

    Examples
    --------
    >>> extractor = JsonTextExtractor()
    >>> extractor.convert_transcript_to_txt()
    >>> print(extractor.get_merged_text())
    """

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.directory_path = os.path.join(script_dir, '../uploaded_files/json')
        self.merged_text = []

    def extract_text_recursive(self, data):
        """
        Recursively extracts text from a JSON data structure.
        Extracts all the text in the json `text` key in the data object and appends it to the `merged_text` list.

        Parameters
        ----------
        data: dict or list
            The JSON data to extract text from.

        Returns
        -------
        None
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "text":
                    self.merged_text.append(str(value))
                elif isinstance(value, (dict, list)):
                    self.extract_text_recursive(value)
        elif isinstance(data, list):
            for item in data:
                self.extract_text_recursive(item)

    def load_and_merge_json_files(self):
        """
        Load and merge multiple JSON files from a directory.

        This method loops through all files in the specified directory and extracts the text from each JSON file.
        It merges the extracted text from all files into a single dictionary.

        Parameters
        ---------
            directory_path (str): The path to the directory containing the JSON files.

        Raises
        ------
            valueError: If the directory path is invalid or does not exist.

        Returns
        -------
            None: The merged JSON data is stored in the object's instance variable.

        Example Usage:
        >>> extractor = JsonTextExtractor()
        >>> extractor.load_and_merge_json_files(directory_path="/path/to/directory")
        """
        if not os.path.exists(self.directory_path) or not os.path.isdir(self.directory_path):
            raise ValueError(f"Invalid directory path: {self.directory_path}")

        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)

            if os.path.isfile(file_path) and filename.lower().endswith('.json'):
                try:
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        self.extract_text_recursive(data)
                except json.JSONDecodeError as e:
                    log.error(f"Error decoding JSON in file {file_path}: {e}")

    def get_merged_text(self):
        """
        Get the merged text.

        This method returns the merged text as a single string.
        It joins the list of merged text elements with a space separator.

        Returns
        -------
        str
            The merged text.
        """
        return " ".join(self.merged_text)

    def convert_transcript_to_txt(self):
        """
        Converts transcript JSON files to a TXT file and saves it in the txt folder.

        Deletes the existing 'txt' folder and creates a new one.
        Loads and merges all JSON files in the 'uploaded_files' directory.
        Write the merged text to a file called 'transcript.txt' in the 'txt' folder.
        Log a message confirming the successful conversion.

        Returns
        -------
        None

        Notes
        -----
        The saved txt file is located in the 'uploaded_files/txt' folder relative to this script
        and is named 'transcript.txt'.
        The 'txt' folder is created if it does not exist.
        The 'txt' folder is deleted and recreated if it already exists to avoid appending to an existing file.
        The 'txt' folder is created in the same directory as this script.
        """
        # Delete the txt folder if it exists and create a new one
        shutil.rmtree('./uploaded_files/txt') if os.path.isdir('./uploaded_files/txt') else None
        os.makedirs('./uploaded_files/txt')

        self.load_and_merge_json_files()

        with open('uploaded_files/txt/transcript.txt', 'w') as output_file:
            output_file.write(self.get_merged_text())

        log.info('Transcript saved as txt file')
