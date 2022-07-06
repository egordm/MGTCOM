source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM --eval_inference=true --split_force=true "
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_DBLP="--batch_size=128 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_IMDB="--batch_size=128 --max_epochs=50 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_ICEWS="--batch_size=128 --max_epochs=50 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_SDS="--batch_size=2048 --max_epochs=40 --dataset=SocialDistancingStudents --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"

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

EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=ablation_inference $ARGS_BASE --lr=0.02 --repr_dim=32 --num_workers=3 --metric=DOTP"

ARGS_DS="$ARGS_DBLP"
EMBED_DS="$EMBED_DBLP"
#EMBED_FULL_DS="$EMBED_DBLP"
$(echo $EXPERIMENT) --run_name="embed0.8" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.1 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.9" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.001 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.7" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.2 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.6" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.3 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.5" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.4 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.4" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.5 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.3" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.6 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.2" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.7 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.1" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.8 --split_num_test=0.1



EXPERIMENT="$ARGS_CMD/mgcom_combi_executor.py --experiment=ablation_inference $ARGS_BASE --lr=0.02 --topo_repr_dim=32 --tempo_repr_dim=32 --repr_dim=32 --num_workers=3 --metric=DOTP"

ARGS_DS="$ARGS_DBLP"
EMBED_DS="$EMBED_DBLP"
#EMBED_FULL_DS="$EMBED_DBLP"
$(echo $EXPERIMENT) --run_name="embed0.8" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.1 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.9" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.001 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.7" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.2 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.6" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.3 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.5" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.4 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.4" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.5 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.3" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.6 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.2" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.7 --split_num_test=0.1
$(echo $EXPERIMENT) --run_name="embed0.1" $(echo "$ARGS_DS $EMBED_DS") --split_num_val=0.8 --split_num_test=0.1

