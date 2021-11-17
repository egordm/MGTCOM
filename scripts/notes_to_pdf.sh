 #!/bin/sh
 
echo "Converting Notes to PDF"
for f in ./notes/**/*.md; do
    if [[ ! -f ${f/md/pdf} || "${f}" -nt "${f/md/pdf}" ]]; then
        if grep -q -E "title: .+" "${f}"; then
            if grep -q -E "type: paper" "${f}"; then
                echo "Converting ${f} to pdf";
                RF=$(realpath "$f");
                sh -c "cd ./notes/Meta/Template && ./to_pdf.sh '${RF}'";
            fi;
        fi;
    fi;
done
