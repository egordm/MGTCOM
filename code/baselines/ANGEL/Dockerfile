FROM python:3.9-slim

# Init the structure
WORKDIR /app
RUN mkdir /input && mkdir /output

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the code
COPY . .

ENTRYPOINT ["python", "run.py"]