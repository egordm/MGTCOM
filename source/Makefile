update:
	@echo 'Updating dependencies'
	mamba env update --prefix=./env --f environment.yml

lock:
	@echo 'Locking dependencies'
	mamba env export --prefix=./env -f environment.lock.yml

install: update lock