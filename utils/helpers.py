import requests


def has_internet_connection():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
