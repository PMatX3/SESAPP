# Use Python 3.9-slim as the base image for a smaller final image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]