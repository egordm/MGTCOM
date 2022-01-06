FROM python:3.6.1-slim

WORKDIR /app

RUN apt update && apt install -y gcc tk

# Set up the environment
COPY requirements.txt .
RUN pip install -r requirements.txt

# Build app
COPY ./utils/ ./utils/
COPY cython_utils.py .

RUN python cython_utils.py build_ext \
    && cp ./build/**/**/* ./utils

# Copy app
COPY . .

ENTRYPOINT ["python", "main.py"]