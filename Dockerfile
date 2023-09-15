# Use a Python 3.9-slim base image for AR
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose a port to communicate with the Flask app
EXPOSE 8000

# Define the command to run your application
CMD ["python3", "app.py"]

