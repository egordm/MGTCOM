FROM python:3.9-slim

RUN apt update && apt install -y gcc

# Init the structure
WORKDIR /app
RUN mkdir /input && mkdir /output

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the code
COPY . .

ENTRYPOINT ["bash", "run.sh"]