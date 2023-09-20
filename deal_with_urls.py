import re

import requests
from bs4 import BeautifulSoup


def extract_text_from_urls(urls):
    all_content = ''
    for url in urls:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # check if the content type is HTML or xml
            content_type = response.headers.get('Content-Type', '').lower()
            if 'xml' in content_type:
                # Parse the XML content of the page
                soup = BeautifulSoup(response.text, 'xml')
            # elif 'pdf' in content_type:
            # TODO: Extract text from PDF url
            else:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all text within <p> tags or any other relevant HTML tags
            text = ''
            for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'span']):
                text += paragraph.get_text() + ' '

            # remove any extra newlines
            text = re.sub('\n+', '\n', text)

            all_content += text
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)

        return all_content


# # Example usage:
# url = ['https://en.wikipedia.org/wiki/Machine_learning', 'https://en.wikipedia.org/wiki/Artificial_neural_network']
# text = extract_text_from_urls(url)
#
# if text:
#     print(text)
#
# # export the visible_text
# with open('textoo.txt', 'w', encoding='utf-8') as f:
#     f.write(text)
