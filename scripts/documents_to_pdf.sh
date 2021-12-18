#!/bin/sh

SCRIPT_DIR="$(realpath $(dirname "$0"))"

echo "Converting Notes to PDF"
for f in ./documents/Documents/*; do
    echo "$f/meta.yml"
    if [ -f "$f/meta.yml" ]; then
        echo "Converting ${f} to pdf";
        RF="$(realpath "$f")";
        echo $RF;
        sh "$SCRIPT_DIR/pandoc/build_acm.sh" "${RF}";
    fi;
done
