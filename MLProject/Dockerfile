# Use the appropriate Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install git BEFORE installing Python dependencies
RUN apt-get update && \
    apt-get install -y git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file first
COPY requirements.txt requirements.txt

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .
# Commands to run your application when the container starts
CMD ["python", "modelling.py"]