# Use a base image that includes GPU-related libraries and dependencies
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including libGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that your Flask app is listening on
EXPOSE 5000

# Define the command to run your application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

