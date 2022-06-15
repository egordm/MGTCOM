source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_DBLP="--batch_size=128 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"

EMBED_DBLP="--embed_node_types Venue Author"
EMBED_DBLP_FULL="--embed_node_types Venue Author Paper"

REPR_DIM=32
EXPERIMENT="$ARGS_CMD/mgcom_combi_executor.py --experiment=abl_beta_topo $ARGS_BASE --lr=0.02 --topo_repr_dim=$REPR_DIM --tempo_repr_dim=$REPR_DIM --repr_dim=$REPR_DIM --num_workers=4 --metric=DOTP"

ARGS_DS="$ARGS_DBLP"
EMBED_DS="$EMBED_DBLP"
EMBED_FULL_DS="$EMBED_DBLP"

for i in `seq 1 3`; do
  #$(echo $EXPERIMENT) --run_name="beta_r=0.5" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.5 --tempo_weight=0.5
#  $(echo $EXPERIMENT) --run_name="beta_r=0.25" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.25 --tempo_weight=0.75
  #$(echo $EXPERIMENT) --run_name="beta_r=0.75" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.75 --tempo_weight=0.25
#  $(echo $EXPERIMENT) --run_name="beta_r=0.4" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.4 --tempo_weight=0.6
  #$(echo $EXPERIMENT) --run_name="beta_r=0.6" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.6 --tempo_weight=0.4
  $(echo $EXPERIMENT) --run_name="beta_r=0.15" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.15 --tempo_weight=0.85 --ballroom_params.walks_per_node=5
  $(echo $EXPERIMENT) --run_name="beta_r=0.15" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.15 --tempo_weight=0.85 --ballroom_params.walks_per_node=10
  $(echo $EXPERIMENT) --run_name="beta_r=0.15" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.15 --tempo_weight=0.85 --ballroom_params.walks_per_node=20
  #$(echo $EXPERIMENT) --run_name="beta_r=0.85" $(echo "$ARGS_DS $EMBED_DS") --topo_weight=0.85 --tempo_weight=0.15
done