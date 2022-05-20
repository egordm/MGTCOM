ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_DBLP="--batch_size=128 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_IMDB="--batch_size=128 --max_epochs=50 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_ICEWS="--batch_size=128 --max_epochs=50 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"

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

# Tune Ballroom Walk
EXPERIMENT="$ARGS_CMD/mgcom_tempo_executor.py --experiment=mgtcom_tune_ballroom $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --context_size=20 --run_name="c20"
$(echo $EXPERIMENT) --context_size=10 --run_name="c10"
$(echo $EXPERIMENT) --context_size=5  --run_name="c5"
$(echo $EXPERIMENT) --context_size=3  --run_name="c3"
$(echo $EXPERIMENT) --walks_per_node=10 --context_size=10 --run_name="c10w10"
$(echo $EXPERIMENT) --walks_per_node=20 --context_size=10 --run_name="c10w20"
$(echo $EXPERIMENT) --walks_per_node=5  --context_size=10 --run_name="c10w5"
$(echo $EXPERIMENT) --num_neg_samples=1 --context_size=10 --run_name="c10nns1"
$(echo $EXPERIMENT) --num_neg_samples=2 --context_size=10 --run_name="c10nns2"
$(echo $EXPERIMENT) --num_neg_samples=5 --context_size=10 --run_name="c10nns2"
$(echo $EXPERIMENT) --walk_length=80 --context_size=10 --run_name="c10wl80"
$(echo $EXPERIMENT) --walk_length=40 --context_size=10 --run_name="c10wl40"
$(echo $EXPERIMENT) --walk_length=20 --context_size=10 --run_name="c10wl20"
$(echo $EXPERIMENT) --walk_length=10 --context_size=10 --run_name="c10wl10"


# Tune E2E
EXPERIMENT="$ARGS_CMD/mgcom_e2e_executor.py --experiment=mgtcom_e2e $ARGS_BASE $ARGS_DS --classification_eval.interval=10"
$(echo $EXPERIMENT) --batch_size=200 --max_epochs=200 --prior_sigma_scale=0.05 --use_tempo=false --offline