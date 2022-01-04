SHELL := /bin/bash
PY := source ../activate.sh && python

data/raw/ucidata-zachary:
	cd data/raw \
		&& wget -c http://konect.cc/files/download.tsv.ucidata-zachary.tar.bz2 -O - | tar -xvjf -

data/raw/enron-mail-20150507:
	cd data/raw \
		&& kaggle datasets download wcukierski/enron-email-dataset \
		&& mkdir enron-mail-20150507 \
		&& unzip -p enron-email-dataset.zip emails.csv > enron-mail-20150507/emails.csv \
		&& rm enron-email-dataset.zip

data/raw/misc-football:
	cd data/raw \
		&& wget https://nrvis.com/download/data/misc/misc-football.zip \
		&& unzip -d misc-football misc-football.zip \
		&& rm misc-football.zip

data/raw/imdb-5000-movie-dataset:
	cd data/raw \
		&& kaggle datasets download carolzhangdc/imdb-5000-movie-dataset \
		&& unzip -d imdb-5000-movie-dataset imdb-5000-movie-dataset.zip \
		&& rm imdb-5000-movie-dataset.zip

data/raw/DBLP-V1:
	cd data/raw \
		&& wget https://lfs.aminer.cn/lab-datasets/citation/citation-network1.zip \
		&& unzip -d DBLP-V1 citation-network1.zip \
		&& rm citation-network1.zip

data/raw/DBLP-V3:
	cd data/raw \
		&& wget https://lfs.aminer.cn/lab-datasets/citation/DBLP-citation-network-Oct-19.tar.gz \
		&& mkdir DBLP-V3 \
		&& tar -xf DBLP-citation-network-Oct-19.tar.gz -C DBLP-V3 \
		&& rm DBLP-citation-network-Oct-19.tar.gz

data/raw/DBLP-HCN:
	cd data/raw \
		&& wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/t4xmpbrr6v-1.zip \
		&& unzip -d DBLP-HCN t4xmpbrr6v-1.zip \
		&& cd DBLP-HCN && unrar e dataset.rar \
		&& rm ../t4xmpbrr6v-1.zip dataset.rar

data/raw/social-distancing-student:
	@echo "Dataset social-distancing-student dataset must be downloaded manually."

data/raw/sx-mathoverflow:
	cd data/raw \
		&& wget https://snap.stanford.edu/data/sx-mathoverflow-a2q.txt.gz \
		&& wget https://snap.stanford.edu/data/sx-mathoverflow-c2q.txt.gz \
		&& wget https://snap.stanford.edu/data/sx-mathoverflow-c2a.txt.gz \
		&& mkdir sx-mathoverflow \
		&& gunzip -c sx-mathoverflow-a2q.txt.gz > sx-mathoverflow/sx-mathoverflow-a2q.txt \
		&& gunzip -c sx-mathoverflow-c2q.txt.gz > sx-mathoverflow/sx-mathoverflow-c2q.txt \
		&& gunzip -c sx-mathoverflow-c2a.txt.gz > sx-mathoverflow/sx-mathoverflow-c2a.txt \
		&& rm sx-mathoverflow-a2q.txt.gz sx-mathoverflow-c2q.txt.gz sx-mathoverflow-c2a.txt.gz

data/raw/sx-superuser:
	cd data/raw \
   		&& wget https://snap.stanford.edu/data/sx-superuser-a2q.txt.gz \
   		&& wget https://snap.stanford.edu/data/sx-superuser-c2q.txt.gz \
   		&& wget https://snap.stanford.edu/data/sx-superuser-c2a.txt.gz \
   		&& mkdir sx-superuser \
   		&& gunzip -c sx-superuser-a2q.txt.gz > sx-superuser/sx-superuser-a2q.txt \
   		&& gunzip -c sx-superuser-c2q.txt.gz > sx-superuser/sx-superuser-c2q.txt \
   		&& gunzip -c sx-superuser-c2a.txt.gz > sx-superuser/sx-superuser-c2a.txt \
   		&& rm sx-superuser-a2q.txt.gz sx-superuser-c2q.txt.gz sx-superuser-c2a.txt.gz

data/raw/email-Eu-core:
	cd data/raw \
		&& wget https://snap.stanford.edu/data/email-Eu-core.txt.gz \
		&& wget https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz \
		&& mkdir email-Eu-core \
		&& gunzip -c email-Eu-core.txt.gz > email-Eu-core/email-Eu-core.txt \
		&& gunzip -c email-Eu-core-department-labels.txt.gz > email-Eu-core/email-Eu-core-department-labels.txt \
		&& rm email-Eu-core.txt.gz email-Eu-core-department-labels.txt.gz


data/raw/com-youtube:
	cd data/raw \
		&& wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz \
		&& wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz \
		&& wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz \
		&& mkdir com-youtube \
		&& gunzip -c com-youtube.ungraph.txt.gz > com-youtube/com-youtube.ungraph.txt \
		&& gunzip -c com-youtube.all.cmty.txt.gz > com-youtube.all.cmty.txt \
		&& gunzip -c com-youtube.top5000.cmty.txt.gz > com-youtube/com-youtube.top5000.cmty.txt \
		&& rm com-youtube.ungraph.txt.gz com-youtube.all.cmty.txt.gz com-youtube.top5000.cmty.txt.gz


download: \
	data/raw/ucidata-zachary \
	data/raw/enron-mail-20150507 \
	data/raw/misc-football \
	data/raw/imdb-5000-movie-dataset \
	data/raw/DBLP-V1 \
	data/raw/DBLP-V3 \
	data/raw/DBLP-HCN \
	data/raw/sx-mathoverflow \
	data/raw/sx-superuser \
	data/raw/email-Eu-core \
	data/raw/com-youtube

export_neo4j:
	$(PY) scripts/export_to_neo4j.py --dataset=ucidata-zachary
	$(PY) scripts/export_to_neo4j.py --dataset=enron-mail-20150507
	$(PY) scripts/export_to_neo4j.py --dataset=misc-football
	$(PY) scripts/export_to_neo4j.py --dataset=imdb-5000-movie-dataset
	$(PY) scripts/export_to_neo4j.py --dataset=DBLP-V1
	$(PY) scripts/export_to_neo4j.py --dataset=DBLP-V3
	$(PY) scripts/export_to_neo4j.py --dataset=DBLP-HCN
	$(PY) scripts/export_to_neo4j.py --dataset=sx-mathoverflow
	$(PY) scripts/export_to_neo4j.py --dataset=sx-superuser
	$(PY) scripts/export_to_neo4j.py --dataset=email-Eu-core
	$(PY) scripts/export_to_neo4j.py --dataset=com-youtube

export_graphml:
	$(PY) scripts/export_to_graphml.py --dataset=ucidata-zachary
	$(PY) scripts/export_to_graphml.py --dataset=enron-mail-20150507
	$(PY) scripts/export_to_graphml.py --dataset=misc-football
	$(PY) scripts/export_to_graphml.py --dataset=imdb-5000-movie-dataset
	$(PY) scripts/export_to_graphml.py --dataset=DBLP-V1
	$(PY) scripts/export_to_graphml.py --dataset=DBLP-V3
	$(PY) scripts/export_to_graphml.py --dataset=DBLP-HCN
	$(PY) scripts/export_to_graphml.py --dataset=sx-mathoverflow
	$(PY) scripts/export_to_graphml.py --dataset=sx-superuser
	$(PY) scripts/export_to_graphml.py --dataset=email-Eu-core
	$(PY) scripts/export_to_graphml.py --dataset=com-youtube