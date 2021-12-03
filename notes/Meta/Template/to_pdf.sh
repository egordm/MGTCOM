#!/bin/sh

SCRIPT_PATH="$(realpath $(dirname "$0"))"
INPUT_FILE="$(realpath "$1")"
WORKDIR="$(dirname $INPUT_FILE)"
OUTPUT_FILE="${INPUT_FILE/md/pdf}"

SOURCE_FORMAT="markdown\
+pipe_tables\
+backtick_code_blocks\
+strikeout\
+yaml_metadata_block\
+implicit_figures\
+all_symbols_escapable\
+link_attributes\
+smart\
+fenced_divs"

PANDOC_PREFIX="$WORKDIR" pandoc -s -o "$OUTPUT_FILE" -t latex -i "$INPUT_FILE" --pdf-engine=latexmk -f "$SOURCE_FORMAT" --natbib --number-sections --template="${SCRIPT_PATH}/main.tex" --filter pandoc-xnos --bibliography="refs.bib" --filter="${SCRIPT_PATH}/uphead" --resource-path="$WORKDIR"

