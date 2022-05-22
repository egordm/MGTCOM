source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_DBLP="--batch_size=128 --max_epochs=60 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_IMDB="--batch_size=128 --max_epochs=60 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_ICEWS="--batch_size=128 --max_epochs=40 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_SDS="--batch_size=2048 --max_epochs=30 --dataset=SocialDistancingStudents --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"

# Embeddings
EMBED_SW="--embed_node_types Character"
EMBED_CORA="--embed_node_types 0"

EMBED_DBLP="--embed_node_types Venue Author"
EMBED_DBLP_FULL="--embed_node_types Venue Author Paper"

EMBED_IMDB="--embed_node_types Genre Person"
EMBED_IMDB_FULL="--embed_node_types Genre Person Movie"

EMBED_SDS="--embed_node_types Hashtag User"
EMBED_SDS_FULL="--embed_node_types Hashtag User Tweet"

EMBED_ICEWS="--embed_node_types Entity"

EXPERIMENT="$ARGS_CMD/mgcom_tempo_executor.py --experiment=benchmark_mgtcom_tempo $ARGS_BASE --lr=0.02 --repr_dim=32 --num_workers=3 --metric=DOTP"
#EXPERIMENT="$ARGS_CMD/mgcom_tempo_executor.py --experiment=benchmark_mgtcom_tempo $ARGS_BASE --lr=0.0005 --repr_dim=64 --num_workers=3 --metric=DOTP"


#ARGS_DS="$ARGS_DBLP"
#EMBED_DS="$EMBED_DBLP"
#EMBED_FULL_DS="$EMBED_DBLP"
##$(echo $EXPERIMENT) --run_name="feat" $(echo $ARGS_DS)
#$(echo $EXPERIMENT) --run_name="embed" $(echo "$ARGS_DS $EMBED_DS")
##$(echo $EXPERIMENT) --run_name="embed_full" $(echo "$ARGS_DS $EMBED_FULL_DS")
#
#ARGS_DS="$ARGS_IMDB"
#EMBED_DS="$EMBED_IMDB"
#EMBED_FULL_DS="$EMBED_IMDB_FULL"
##$(echo $EXPERIMENT) --run_name="feat" $(echo $ARGS_DS)
#$(echo $EXPERIMENT) --run_name="embed" $(echo "$ARGS_DS $EMBED_DS")
##$(echo $EXPERIMENT) --run_name="embed_full" $(echo "$ARGS_DS $EMBED_FULL_DS")
#
#ARGS_DS="$ARGS_ICEWS"
#EMBED_DS="$EMBED_ICEWS"
##$(echo $EXPERIMENT) --run_name="feat" $(echo $ARGS_DS)
#$(echo $EXPERIMENT) --run_name="embed" $(echo "$ARGS_DS $EMBED_DS")

ARGS_DS="$ARGS_SDS"
EMBED_DS="$EMBED_SDS"
#$(echo $EXPERIMENT) --run_name="feat" $(echo $ARGS_DS)
$(echo $EXPERIMENT) --run_name="embed" $(echo "$ARGS_DS $EMBED_DS")