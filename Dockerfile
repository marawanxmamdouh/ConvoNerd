# Use the Python 3.11.2 slim image as the base for this container.
FROM python:3.11.2-slim

# Add author metadata.
LABEL authors="Marawan Mamdouh"

# Set the working directory within the container to /app.
WORKDIR /app

# Update the package index and install necessary packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libmagic-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Clone the ConvoNerd repository.
RUN git clone https://github.com/marawanxmamdouh/ConvoNerd.git

# Change the working directory to the cloned repository.
WORKDIR /app/ConvoNerd

# Upgrade pip to the latest version.
RUN pip install --upgrade pip

# Install project dependencies
RUN pip install -r requirements.txt

# Expose port 8501 for external access.
EXPOSE 8501

# Set the default command to runa health check for the container.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Define the entry point command to run the Streamlit application.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]
