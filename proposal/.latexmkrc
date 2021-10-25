$pdf_mode = 1;
$bibtex_use = 1;
$force_mode = 1;

$out_dir = '.';

$pdflatex = 'pdflatex -file-line-error -interaction=batchmode -synctex=1';
$bibtex   = "bibtex %S";
$pdf_previewer="start okular %O %S";
system("echo 'Copying Refs' && cp ../refs.bib refs.bib");
@default_files = ('main.tex');
