# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app

# Expose the port that your FastAPI application will listen on
EXPOSE 8000

# Define the command to run your FastAPI application
CMD ["uvicorn", "app.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]