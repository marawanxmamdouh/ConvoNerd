import os
import shutil

import requests
from loguru import logger as log


def has_internet_connection():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def save_uploaded_files(uploaded_files):
    base_path = 'uploaded_files'

    # Delete the base_path folder if it exists
    if os.path.isdir(base_path):
        shutil.rmtree(base_path)

    # Create the base folder if it does not exist
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    for uploaded_file in uploaded_files:
        target_folder = prepare_target_folder(base_path, uploaded_file.name)

        # Save the uploaded file to the target folder
        save_file_to_folder(uploaded_file, target_folder)

        # show a success message for 2 seconds and then hide it.
        log.success(f"File uploaded successfully: {uploaded_file.name}", 2)


def prepare_target_folder(base_path, filename):
    # Create the sub-folder for the file type if it doesn't exist
    file_extension = os.path.splitext(filename)[1]
    target_folder = os.path.join(base_path, file_extension.lstrip('.'))

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    return target_folder


def save_file_to_folder(file_object, target_folder):
    target_file_path = os.path.join(target_folder, file_object.name)
    with open(target_file_path, 'wb') as f:
        f.write(file_object.getvalue())
