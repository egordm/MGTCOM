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

# Tune E2E
EXPERIMENT="$ARGS_CMD/mgcom_e2e_executor.py --experiment=mgtcome2e_tume_feat $ARGS_BASE"
$(echo $EXPERIMENT) --prior_sigma_scale=0.05 --use_tempo=false --name="feat" $(echo $ARGS_DS)
$(echo $EXPERIMENT) --prior_sigma_scale=0.05 --use_tempo=false --name="embed" $(echo $EMBED_DS)
$(echo $EXPERIMENT) --prior_sigma_scale=0.05 --use_tempo=false --name="embed_full" $(echo $EMBED_FULL_DS)
