#!/bin/bash

SERVE_DIR="$(realpath "$(dirname "$0")")"

python -m http.server 8080 --bind 127.0.0.1 --directory "$SERVE_DIR"