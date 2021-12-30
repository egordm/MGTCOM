update:
	@echo 'Updating dependencies'
	conda env update --prefix ./env --file environment.yml --prune

lock:
	@echo 'Locking dependencies'
	conda env export --prefix=./env -f environment.yml.lock

install: update lock

activate:
	conda activate ./env