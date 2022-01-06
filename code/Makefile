update:
	@echo 'Updating dependencies'
	conda env update --prefix=./env --f environment.yml --prune

lock:
	@echo 'Locking dependencies'
	conda env export --prefix=./env -f environment.yml.lock

install: update lock