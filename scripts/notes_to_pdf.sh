#!/bin/sh

SCRIPT_DIR="$(realpath $(dirname "$0"))"

echo "Converting Notes to PDF"
for f in ./documents/**/*.md; do
    if [[ ! -f ${f/md/pdf} || "${f}" -nt "${f/md/pdf}" ]]; then
        if grep -q -E "title: .+" "${f}"; then
            if grep -q -E "type: paper" "${f}"; then
                echo "Converting ${f} to pdf";
                RF="$(realpath "$f")";
                echo $RF;
                sh "$SCRIPT_DIR/pandoc/build_acm.sh" "${RF}";
            fi;
        fi;
    fi;
done
