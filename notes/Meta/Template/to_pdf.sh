#!/bin/sh

SCRIPT_PATH="$(realpath $(dirname "$0"))"
INPUT_FILE="$(realpath "$1")"
WORKDIR="$(dirname $INPUT_FILE)"
OUTPUT_FILE="${INPUT_FILE/md/pdf}"

PANDOC_PREFIX="$WORKDIR" pandoc -s -o "$OUTPUT_FILE" -t latex -i "$INPUT_FILE" --pdf-engine=xelatex --template="${SCRIPT_PATH}/main.tex" --filter="${SCRIPT_PATH}/uphead" --resource-path="$WORKDIR"
