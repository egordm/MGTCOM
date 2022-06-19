source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=20"
ARGS_CORA="--batch_size=128 --max_epochs=200 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=20"
ARGS_DBLP="--batch_size=128 --max_epochs=200 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=20"
ARGS_IMDB="--batch_size=128 --max_epochs=200 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=20"
ARGS_ICEWS="--batch_size=128 --max_epochs=200 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=20"

EMBED_SW="--embed_node_types Character"
EMBED_CORA="--embed_node_types 0"
EMBED_DBLP="--embed_node_types Venue Author"
EMBED_DBLP_FULL="--embed_node_types Venue Author Paper"
EMBED_IMDB="--embed_node_types Genre Person"
EMBED_IMDB_FULL="--embed_node_types Genre Person Movie"

#ARGS_DS="$ARGS_CORA"
#EMBED_DS="$EMBED_CORA"
#EMBED_FULL_DS="$EMBED_CORA"

ARGS_DS="$ARGS_DBLP"
EMBED_DS="$EMBED_DBLP"
EMBED_FULL_DS="$EMBED_DBLP"

#ARGS_DS="$ARGS_IMDB"
#EMBED_DS="$EMBED_IMDB"
#EMBED_FULL_DS="EMBED_IMDB_FULL"

#ARGS_DS="$ARGS_ICEWS"
#EMBED_DS=""
#EMBED_FULL_DS=""
