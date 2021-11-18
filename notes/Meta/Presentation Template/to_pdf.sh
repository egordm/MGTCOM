#!/usr/bin/sh

DATE_COVER=$(date "+%d %B %Y")

SOURCE_FORMAT="markdown\
+pipe_tables\
+backtick_code_blocks\
+auto_identifiers\
+strikeout\
+yaml_metadata_block\
+implicit_figures\
+all_symbols_escapable\
+link_attributes\
+smart\
+fenced_divs"
# +citations\

DATA_DIR="pandoc"

SCRIPT_PATH="$(realpath $(dirname "$0"))"
INPUT_FILE="$(realpath "$1")"
WORKDIR="$(dirname $INPUT_FILE)"
OUTPUT_FILE="${INPUT_FILE/md/pdf}"

PANDOC_PREFIX="$WORKDIR"  pandoc -s --dpi=300 --slide-level 2 --toc --natbib --listings --shift-heading-level=0 --data-dir="${DATA_DIR}" --template "${SCRIPT_PATH}/template.tex" -H "${SCRIPT_PATH}/preamble.tex" --pdf-engine latexmk -f "$SOURCE_FORMAT" -M date="$DATE_COVER" -V classoption=notes -V classoption:aspectratio=169 -t beamer -i "$INPUT_FILE" -o "$OUTPUT_FILE" --bibliography="refs.bib" --filter="${SCRIPT_PATH}/uphead" --resource-path="$WORKDIR"

# 
