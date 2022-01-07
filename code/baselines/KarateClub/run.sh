#!/bin/sh

SCRIPT_DIR="$(realpath $(dirname $0))"
python "$SCRIPT_DIR/$1.py" ${@:2}