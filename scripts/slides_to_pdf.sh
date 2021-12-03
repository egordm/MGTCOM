 #!/bin/sh
 
echo "Converting Slides to PDF"
for f in ./documents/**/*.md; do
    if [[ ! -f ${f/md/pdf} || "${f}" -nt "${f/md/pdf}" ]]; then
        if grep -q -E "title: .+" "${f}"; then
            if grep -q -E "type: slides" "${f}"; then
                echo "Converting ${f} to pdf";
                RF=$(realpath "$f");
                sh -c "cd './notes/Meta/Presentation Template' && ./to_pdf.sh '${RF}'";
            fi;
        fi;
    fi;
done
