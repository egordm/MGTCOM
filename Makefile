update:
	@echo 'Updating dependencies'
	mamba env update --prefix=./env --f environment.yml --prune

lock:
	@echo 'Locking dependencies'
	mamba env export --prefix=./env --no-builds -f environment.lock.yml

install: update lock