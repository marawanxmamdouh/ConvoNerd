# ConvoNerd
AI Chat Assistant is an open-source tool that enables natural language conversations with a wide range of data sources, 
from documents to web links, text data and even YouTube videos. With the power of state-of-the-art language models, 
including Retrieval-Augmented Generation (RAG), this tool empowers you to ask questions, extract insights, 
and explore your data interactively. Enjoy the convenience of a user-friendly interface and the flexibility 
to choose your language model, all while running efficiently on standard CPU hardware.

![ConvoNerd Demo](media/demo.mp4)


## Table of Contents

- [Key Features](#key-features)
- [To Do](#to-do)
- [CPU Optimization](#cpu-optimization)
- [Get Started](#get-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
      - [Using Poetry](#i-using-poetry)
      - [Using pip](#ii-using-pip)
      - [Using Docker](#iii-using-docker)
      - [Using Colab GPU](#iv-using-colab-with-gpu-runtime)
    - [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Key Features

- **Versatile Data Sources:** AI Chat Assistant provides flexibility in interacting with your data through various sources:
    1. **Uploaded Documents:** Easily converse with content from your uploaded documents, including PDFs, txt files, and markdown files.
    2. **Web Links:** Engage in conversations with data from web pages, news articles, and online sources.
    3. **Manual Text Input:** Input text directly to inquire and discuss specific information or ideas.
    4. **YouTube Video Links:** Interact with YouTube videos to gain insights from video content or ask questions about the video.
- **Choice of Language Models:** Tailor your conversations with a selection of language models. Opt for CPU-based models or GPU-based models, depending on your device's capabilities. don't be limited to high-end GPUs, you can use your CPU to chat with your data.
- **Conversation History:** Stay organized by tracking your entire chat history with the chat history object. Review and analyze past interactions with ease. and it remembers the context of your last conversation, making it effortless to ask follow-up questions without mentioning entities.
- **Clear and User-Friendly Interface:** The user interface is designed for clarity and ease of use, ensuring a smooth experience for all users.
- **Open-Source and Customizable:** AI Chat Assistant is an open-source project, making it customizable and extensible. Integrate it into your projects, enhance its functionality, and contribute to its development. Your input is valued and welcomed.

## CPU Optimization

One of our primary objectives is to make AI Chat Assistant accessible to a wide range of users. While some language models demand GPU resources, we've optimized this project to efficiently run on CPU. This means you can use AI Chat Assistant on most standard hardware configurations, eliminating the need for specialized hardware, making it more accessible and cost-effective.

By providing an implementation of RAG from scratch and ensuring CPU compatibility, AI Chat Assistant offers a robust and accessible solution for engaging in meaningful conversations with your data.


## Get Started

### Prerequisites

- [Python](https://www.python.org/downloads/) (tested on version 3.11.2)
- Language Model to download check [Model Download Instructions](models/Model Download Instructions.md)

### Installation

#### **I. Using Poetry**

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/marawanxmamdouh/ConvoNerd.git
    ```

1. Change the working directory to the project folder:

    ```shell
    cd ConvoNerd
    ```

2. Install [Poetry](https://python-poetry.org) if you haven't already:

    ```shell
    pip install poetry
    ```

3. Use Poetry to install the project dependencies from the `pyproject.toml` file:

    ```shell
    poetry install
    ```

4. Activate the virtual environment created by Poetry (this step may vary depending on your shell):
    - On Windows:

        ```shell
        poetry shell
        ```

    - On Unix-based systems (Linux/macOS):
        ```shell
        source $(poetry env info --path)/bin/activate
        ```

#### **II. Using PIP**

- If you don't want to use Poetry, you can install the dependencies using a **`requirements.txt`** file instead,
  but I highly recommend using Poetry.

- Note that this method will not create a virtual environment for you, so you will need to create one manually if you
want to avoid polluting your global Python environment.

    ```bash
    pip install -r requirements.txt
    ```

#### **III. Using Docker**

- If you prefer to use Docker, you can build a Docker image from the provided `Dockerfile`:

    ```bash
    docker build -t convo_nerd:latest .
    ```

    Then you can run the Docker image in a container:
    
    ```bash
    docker run -p 8080:8501 convo_nerd:latest .
    ```
- Or **alternatively**, you can pull the Docker image from Docker Hub:

    ```bash
    docker pull marawanxmamdouh/convo_nerd
    ```

    Finally, you can access the ConvoNerd application by opening [http://localhost:8080](http://localhost:8080/) in your
    web browser.

#### **IV. Using Colab with GPU runtime**

- You can also run ConvoNerd on Google Colab using a GPU runtime to benefit from faster inference times. 
To do so, you can run the attached notebook [ConvoNerd.ipynb](https://colab.research.google.com/github/marawanxmamdouh/ConvoNerd/blob/master/convonerd_colab.ipynb)
on Colab GPU. and follow the instructions in the notebook.

### **Usage**

Now that you have set up ConvoNerd, you can run the application by executing:

```bash
streamlit run app.py
```

This will start a local Streamlit server and launch the ConvoNerd application
in your default web browser by open [http://localhost:8501](http://localhost:8501/) to access the ConvoNerd application.

## Pipeline Simple Overview:

### **Data Source**

- **Data Source Definition**: This is the origin of the data you wish to engage with, which may take the form of a document, web link, or YouTube video.
- **Data Extraction**: The process of extracting raw text from the data source, involving tools like PDF extractors, web scrapers, or YouTube video extractors.
- **Text Processing**: After extraction, raw text undergoes processing, which can include tasks like text cleaning or normalization.
- **Text Chunking**: Text is split into manageable chunks, either as text segments or smaller documents, depending on the input type.
- **Text Encoding**: Text chunks are encoded into embeddings for further analysis.
- **Vector Indexing**: Text chunk embeddings are indexed for efficient retrieval and search.

### **Question Answering**

- **Question Encoding**: The question is transformed into an embedding for compatibility with the model.
- **Vector Retrieval**: The system retrieves the most relevant text chunk embeddings based on the question embedding.
- **Text Selection**: From the retrieved text chunks, the most pertinent ones are selected in response to the question.
- **Memory Creation**: A memory system stores user questions and their respective answers, preserving conversation context.
- **Conversation Chain**: The conversation chain combines questions and memory to maintain context.
- **Language Model Initialization**: The language model is set up to generate answers within the conversation chain.
- **Prompt Generation**: A prompt is created with the question and the relevant text chunks.
- **Answering the Question**: The language model generates an answer in response to the prompt within the conversation chain.
- **Memory Update**: The memory is updated with the question and its corresponding answer.
- **Subsequent Questions**: If the user asks another question, a follow-up question is initially generated based on the previous memory. The process then continues from step 7.

## To Do
- [ ] Support other document types (e.g., docx, pptx, etc.)
- [ ] Support Rust for faster performance
- [ ] Support agents

## **Contributing**

We welcome contributions from the community. If you have ideas for improvements, bug fixes, or suggestions, please
consider contributing to the project.

## **License**

ConvoNerd is licensed under the [MIT License](./LICENSE).

## **Contact**

If you have questions or feedback, feel free to [open an issue](https://github.com/marawanxmamdouh/ConvoNerd/issues) on
this repository.

We hope ConvoNerd empowers you to have meaningful conversations with your data. Enjoy exploring and enhancing your
data-driven insights!
