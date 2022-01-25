SHELL := /bin/bash
PY := source ../activate.sh && python
CONFIG_DIR := ../config/datasets
DATASETS_DIR := ../storage/datasets/raw

../storage/datasets/raw/ucidata-zachary:
	cd ${DATASETS_DIR} \
		&& wget -c http://konect.cc/files/download.tsv.ucidata-zachary.tar.bz2 -O - | tar -xvjf - \
		&& wget -c https://github.com/GiulioRossetti/cdlib_datasets/raw/main/networks/karate_club.csv.gz -O - | gunzip -c > ucidata-zachary/karate_club.csv \
		&& wget -c https://github.com/GiulioRossetti/cdlib_datasets/raw/main/ground_truth/karate_club.json.gz -O - | gunzip -c > ucidata-zachary/karate_club.json \

../storage/datasets/raw/enron-mail-20150507:
	cd ${DATASETS_DIR} \
		&& kaggle datasets download wcukierski/enron-email-dataset \
		&& mkdir enron-mail-20150507 \
		&& unzip -p enron-email-dataset.zip emails.csv > enron-mail-20150507/emails.csv \
		&& rm enron-email-dataset.zip

../storage/datasets/raw/misc-football:
	cd ${DATASETS_DIR} \
		&& wget https://nrvis.com/download/data/misc/misc-football.zip \
		&& unzip -d misc-football misc-football.zip \
		&& rm misc-football.zip

../storage/datasets/raw/imdb-5000-movie-dataset:
	cd ${DATASETS_DIR} \
		&& kaggle datasets download carolzhangdc/imdb-5000-movie-dataset \
		&& unzip -d imdb-5000-movie-dataset imdb-5000-movie-dataset.zip \
		&& rm imdb-5000-movie-dataset.zip

../storage/datasets/raw/DBLP-V1:
	cd ${DATASETS_DIR} \
		&& wget https://lfs.aminer.cn/lab-datasets/citation/citation-network1.zip \
		&& unzip -d DBLP-V1 citation-network1.zip \
		&& rm citation-network1.zip

../storage/datasets/raw/DBLP-V3:
	cd ${DATASETS_DIR} \
		&& wget https://lfs.aminer.cn/lab-datasets/citation/DBLP-citation-network-Oct-19.tar.gz \
		&& mkdir DBLP-V3 \
		&& tar -xf DBLP-citation-network-Oct-19.tar.gz -C DBLP-V3 \
		&& rm DBLP-citation-network-Oct-19.tar.gz

../storage/datasets/raw/DBLP-HCN:
	cd ${DATASETS_DIR} \
		&& wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/t4xmpbrr6v-1.zip \
		&& unzip -d DBLP-HCN t4xmpbrr6v-1.zip \
		&& cd DBLP-HCN && unrar e dataset.rar \
		&& rm ../t4xmpbrr6v-1.zip dataset.rar

../storage/datasets/raw/social-distancing-student:
	@echo "Dataset social-distancing-student dataset must be downloaded manually."

../storage/datasets/raw/sx-mathoverflow:
	cd ${DATASETS_DIR} \
		&& wget https://snap.stanford.edu/data/sx-mathoverflow-a2q.txt.gz \
		&& wget https://snap.stanford.edu/data/sx-mathoverflow-c2q.txt.gz \
		&& wget https://snap.stanford.edu/data/sx-mathoverflow-c2a.txt.gz \
		&& mkdir sx-mathoverflow \
		&& gunzip -c sx-mathoverflow-a2q.txt.gz > sx-mathoverflow/sx-mathoverflow-a2q.txt \
		&& gunzip -c sx-mathoverflow-c2q.txt.gz > sx-mathoverflow/sx-mathoverflow-c2q.txt \
		&& gunzip -c sx-mathoverflow-c2a.txt.gz > sx-mathoverflow/sx-mathoverflow-c2a.txt \
		&& rm sx-mathoverflow-a2q.txt.gz sx-mathoverflow-c2q.txt.gz sx-mathoverflow-c2a.txt.gz

../storage/datasets/raw/sx-superuser:
	cd ${DATASETS_DIR} \
   		&& wget https://snap.stanford.edu/data/sx-superuser-a2q.txt.gz \
   		&& wget https://snap.stanford.edu/data/sx-superuser-c2q.txt.gz \
   		&& wget https://snap.stanford.edu/data/sx-superuser-c2a.txt.gz \
   		&& mkdir sx-superuser \
   		&& gunzip -c sx-superuser-a2q.txt.gz > sx-superuser/sx-superuser-a2q.txt \
   		&& gunzip -c sx-superuser-c2q.txt.gz > sx-superuser/sx-superuser-c2q.txt \
   		&& gunzip -c sx-superuser-c2a.txt.gz > sx-superuser/sx-superuser-c2a.txt \
   		&& rm sx-superuser-a2q.txt.gz sx-superuser-c2q.txt.gz sx-superuser-c2a.txt.gz

../storage/datasets/raw/email-Eu-core:
	cd ${DATASETS_DIR} \
		&& wget https://snap.stanford.edu/data/email-Eu-core.txt.gz \
		&& wget https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz \
		&& mkdir email-Eu-core \
		&& gunzip -c email-Eu-core.txt.gz > email-Eu-core/email-Eu-core.txt \
		&& gunzip -c email-Eu-core-department-labels.txt.gz > email-Eu-core/email-Eu-core-department-labels.txt \
		&& rm email-Eu-core.txt.gz email-Eu-core-department-labels.txt.gz


../storage/datasets/raw/com-youtube:
	cd ${DATASETS_DIR} \
		&& wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz \
		&& wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz \
		&& wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.top5000.cmty.txt.gz \
		&& mkdir com-youtube \
		&& gunzip -c com-youtube.ungraph.txt.gz > com-youtube/com-youtube.ungraph.txt \
		&& gunzip -c com-youtube.all.cmty.txt.gz > com-youtube.all.cmty.txt \
		&& gunzip -c com-youtube.top5000.cmty.txt.gz > com-youtube/com-youtube.top5000.cmty.txt \
		&& rm com-youtube.ungraph.txt.gz com-youtube.all.cmty.txt.gz com-youtube.top5000.cmty.txt.gz

../storage/datasets/raw/the-marvel-universe-social-network:
	cd ${DATASETS_DIR} \
		&& kaggle datasets download csanhueza/the-marvel-universe-social-network \
		&& unzip -d the-marvel-universe-social-network the-marvel-universe-social-network.zip \
		&& rm the-marvel-universe-social-network.zip

../storage/datasets/raw/star-wars:
	cd ${DATASETS_DIR} \
		&& kaggle datasets download ruchi798/star-wars \
		&& unzip -d star-wars star-wars.zip \
		&& rm star-wars.zip

../storage/datasets/raw/house-of-representatives-congress-116:
	cd ${DATASETS_DIR} \
		&& kaggle datasets download aavigan/house-of-representatives-congress-116 \
		&& unzip -d house-of-representatives-congress-116 house-of-representatives-congress-116.zip \
		&& rm house-of-representatives-congress-116.zip

download: \
	../storage/datasets/raw/ucidata-zachary \
	../storage/datasets/raw/enron-mail-20150507 \
	../storage/datasets/raw/misc-football \
	../storage/datasets/raw/imdb-5000-movie-dataset \
	../storage/datasets/raw/DBLP-V1 \
	../storage/datasets/raw/DBLP-V3 \
	../storage/datasets/raw/DBLP-HCN \
	../storage/datasets/raw/sx-mathoverflow \
	../storage/datasets/raw/sx-superuser \
	../storage/datasets/raw/email-Eu-core \
	../storage/datasets/raw/com-youtube \
	../storage/datasets/raw/star-wars \
	../storage/datasets/raw/house-of-representatives-congress-116

export_neo4j:
	for target in $(CONFIG_DIR)/* ; do \
  		$(PY) datasets/scripts/export_to_neo4j.py $$(basename -s .yaml $$target); \
  	done

export_graphml:
	@for target in $(CONFIG_DIR)/* ; do \
  		$(PY) datasets/scripts/export_to_graphml.py $$(basename -s .yaml $$target); \
  	done

version:
	@for target in $(CONFIG_DIR)/* ; do \
  		$(PY) datasets/scripts/export_to_version.py $$(basename -s .yaml $$target); \
  	done