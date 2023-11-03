# Importing the necessary libraries
import os
import shutil

import requests
import yaml
from box import Box
from loguru import logger as log


def get_config(file_name: str) -> Box:
    """
    Load and return the configuration from the specified file.

    Parameters
    ----------
    file_name: str
        The name of the file to load the configuration from relative to the conf folder.

    Returns
    -------
    Box
        The configuration stored in a Box object.

    Examples
    --------
    >>> config = get_config('example.yaml')
    >>> print(config.database.host)
    ... 'localhost'
    >>> print(config.database.port)
    ... 5432
    """
    with open('conf/' + file_name, 'r', encoding='utf8') as yml_file:
        cfg: Box = Box(yaml.safe_load(yml_file))
    return cfg


def has_internet_connection() -> bool:
    """
    Checks if the system has an active Internet connection by making a request to the Google homepage.

    Returns
    -------
    bool
        True if the system can successfully connect to the internet, otherwise False.
    """
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def prepare_target_folder(base_path: str, filename: str) -> str:
    """
    Prepare the target folder for saving the file by creating sub-folders based on file types.

    Parameters
    ----------
    base_path: str
        The base path where the file is to be saved.
    filename: str
        The name of the file to be saved.

    Returns
    -------
    str
      The path of the target folder.
    """
    # Create the sub-folder for the file type if it doesn't exist
    file_extension: str = os.path.splitext(filename)[1]
    target_folder: str = os.path.join(base_path, file_extension.lstrip('.'))

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    return target_folder


def save_file_to_folder(file_object, target_folder: str) -> None:
    """
    Save the given file object to the specified folder.

    Parameters
    ----------
    file_object: UploadedFile | list[UploadedFile]
        The file object to be saved.
    target_folder: str
        The target folder where the file object is to be saved.

    Returns
    -------
    None
    """
    target_file_path = os.path.join(target_folder, file_object.name)
    with open(target_file_path, 'wb') as f:
        f.write(file_object.getvalue())


# Get the configuration
cfg = get_config('paths.yaml')


def save_uploaded_files(uploaded_files) -> None:
    """
    Saves uploaded files to a specified directory.

    Parameters
    ----------
    uploaded_files: UploadedFile | list[UploadedFile]
        A list of file objects to be saved.

    Returns
    -------
    None
    """
    base_path: str = cfg.uploaded_files_dir_path

    # Delete the base_path folder if it exists
    if os.path.isdir(base_path):
        shutil.rmtree(base_path)

    # Create the base folder if it does not exist
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    for uploaded_file in uploaded_files:
        target_folder: str = prepare_target_folder(base_path, uploaded_file.name)

        # Save the uploaded file to the target folder
        save_file_to_folder(uploaded_file, target_folder)

        # show a success message for 2 seconds and then hide it.
        log.success(f"File uploaded successfully: {uploaded_file.name}", 2)
