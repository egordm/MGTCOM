update:
	@echo 'Updating dependencies'
# 	conda env update --prefix=./env --f environment.yml --prune
	conda env update --prefix=./env --f environment.yml

lock:
	@echo 'Locking dependencies'
	conda env export --prefix=./env -f environment.lock.yml

install: update lock