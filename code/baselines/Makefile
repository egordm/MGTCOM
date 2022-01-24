rebuild:
	for target in ./* ; do \
  		if [ -d $$target ] && [ -f $$target/docker-compose.yml ] ; then \
			echo "Building $$target"; \
			cd "$$target"; docker-compose build; cd ..; \
		fi; \
  	done