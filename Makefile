.PHONY: notes

coffee: pdf sync commit

notes:
	typora documents &

copy-refs:
	cp refs.bib "documents/Documents/Proposal/refs.bib"
	cp refs.bib "documents/Meta/Presentation Template/refs.bib"
	cp refs.bib "documents/refs.bib"

documents-to-pdf: copy-refs
	bash ./scripts/documents_to_pdf.sh

notes-to-pdf: copy-refs
	bash ./scripts/notes_to_pdf.sh

slides-to-pdf: copy-refs
	bash ./scripts/slides_to_pdf.sh

pdf: notes-to-pdf documents-to-pdf slides-to-pdf

sync-to-notion:
	notionsci sync zotero collections e1a32bedcda443deb60e20fc5bc2b2e0
	notionsci sync zotero refs e1a32bedcda443deb60e20fc5bc2b2e0

sync-to-local:
	notionsci sync markdown pages 0642b7dd4fee4acf8e48e45b67faad2b ./references

sync: sync-to-notion sync-to-local

commit:
	git add -A
	git commit -m "Autocommit changes on $(shell date -R)"
	$(MAKE) push

push:
	git push origin master
	git push mirror master

present:
	pympress "$(filter-out $@,$(MAKECMDGOALS))"
# 	pympress $(SLIDES)

activate:
	conda activate INFOMDIS

update:
	git submodule foreach git pull origin master

%:
    @:
