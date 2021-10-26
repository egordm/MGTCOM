.PHONY: notes

notes:
	typora notes

sync-to-notion:
	notionsci sync zotero collections e1a32bedcda443deb60e20fc5bc2b2e0
	notionsci sync zotero refs e1a32bedcda443deb60e20fc5bc2b2e0

sync-to-local:
	notionsci sync markdown pages 0642b7dd4fee4acf8e48e45b67faad2b ./references

sync: sync-to-notion sync-to-local

commit:
	git add -A
	git commit -m "Autocommit changes on $(date)"
	$(MAKE) push

push:
	git push origin master
	git push mirror master