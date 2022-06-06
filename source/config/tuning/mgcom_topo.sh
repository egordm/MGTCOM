source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM --num_workers=3 --metric=DOTP"
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

# Tune Repr dim
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_reprdim3 $ARGS_BASE $ARGS_DS $EMBED_DS --lr=0.02"
#for i in `seq 1 3`; do
#  $(echo $EXPERIMENT) --repr_dim=64 --run_name="d64"
#  $(echo $EXPERIMENT) --repr_dim=32 --run_name="d32"
#  $(echo $EXPERIMENT) --repr_dim=16 --run_name="d16"
#  $(echo $EXPERIMENT) --repr_dim=8  --run_name="d8"
#  $(echo $EXPERIMENT) --repr_dim=4  --run_name="d4"
#  $(echo $EXPERIMENT) --repr_dim=16 --conv_hidden_dim=32 --run_name="d16h32"
#  $(echo $EXPERIMENT) --repr_dim=8  --conv_hidden_dim=16 --run_name="d8h32"
#  $(echo $EXPERIMENT) --repr_dim=4  --conv_hidden_dim=16 --run_name="d4h32"
#done

### Tune LR
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_lr $ARGS_BASE $ARGS_DS $EMBED_DS"
#$(echo $EXPERIMENT) --lr=0.1    --run_name="1e-1"
#$(echo $EXPERIMENT) --lr=0.01   --run_name="1e-2"
#$(echo $EXPERIMENT) --lr=0.02   --run_name="2e-2"
#$(echo $EXPERIMENT) --lr=0.05   --run_name="5e-2"
#$(echo $EXPERIMENT) --lr=0.001  --run_name="1e-3"
#$(echo $EXPERIMENT) --lr=0.0001 --run_name="1e-4"

## Tune Hinge Margin
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_hmargin3 $ARGS_BASE $ARGS_DS $EMBED_DS --lr=0.02"
#for i in `seq 1 3`; do
#  $(echo $EXPERIMENT) --hinge_margin=0.5 --run_name="hmargin=0.5"
#  $(echo $EXPERIMENT) --hinge_margin=1   --run_name="hmargin=1"
#  $(echo $EXPERIMENT) --hinge_margin=2   --run_name="hmargin=2"
#  $(echo $EXPERIMENT) --hinge_margin=4   --run_name="hmargin=4"
#  $(echo $EXPERIMENT) --hinge_margin=8   --run_name="hmargin=8"
#  $(echo $EXPERIMENT) --hinge_margin=16  --run_name="hmargin=16"
#done

## Tune Neighbor Sampling
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_neighbors3 $ARGS_BASE $ARGS_DS $EMBED_DS --lr=0.02"
#for i in `seq 1 3`; do
#  $(echo $EXPERIMENT) --conv_method=HGT  --num_samples 3 2 --run_name="hgt_b32"
#  $(echo $EXPERIMENT) --conv_method=HGT  --num_samples 3 1 --run_name="hgt_b31"
#  $(echo $EXPERIMENT) --conv_method=HGT  --num_samples 2 2 --run_name="hgt_b22"
#  $(echo $EXPERIMENT) --conv_method=HGT  --num_samples 6 2 --run_name="hgt_b62"
#  $(echo $EXPERIMENT) --conv_method=HGT  --num_samples 8 4 --run_name="hgt_b84"
#  $(echo $EXPERIMENT) --conv_method=SAGE --num_samples 3 2 --run_name="sage_b32"
#  $(echo $EXPERIMENT) --conv_method=SAGE --num_samples 3 1 --run_name="sage_b31"
#  $(echo $EXPERIMENT) --conv_method=SAGE --num_samples 2 2 --run_name="sage_b22"
#  $(echo $EXPERIMENT) --conv_method=SAGE --num_samples 6 2 --run_name="sage_b62"
#  $(echo $EXPERIMENT) --conv_method=SAGE --num_samples 8 4 --run_name="sage_b84"
#done

### Tune Random Walk
EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_rw3 $ARGS_BASE $ARGS_DS $EMBED_DS --lr=0.02"
for i in `seq 1 3`; do
  $(echo $EXPERIMENT) --context_size=20 --run_name="c20"
  $(echo $EXPERIMENT) --context_size=10 --run_name="c10"
  $(echo $EXPERIMENT) --context_size=5  --run_name="c5"
  $(echo $EXPERIMENT) --context_size=3  --run_name="c3"
  $(echo $EXPERIMENT) --walks_per_node=30 --context_size=10 --run_name="c10w30"
  $(echo $EXPERIMENT) --walks_per_node=20 --context_size=10 --run_name="c10w20"
  $(echo $EXPERIMENT) --walks_per_node=15 --context_size=10 --run_name="c10w20"
  $(echo $EXPERIMENT) --walks_per_node=10 --context_size=10 --run_name="c10w10"
  $(echo $EXPERIMENT) --walks_per_node=5  --context_size=10 --run_name="c10w5"
  $(echo $EXPERIMENT) --q=0.5  --context_size=10 --run_name="c10q0.5"
  $(echo $EXPERIMENT) --q=0.25 --context_size=10 --run_name="c10q0.25"
  $(echo $EXPERIMENT) --q=0.1  --context_size=10 --run_name="c10q0.1"
  $(echo $EXPERIMENT) --q=1    --context_size=10 --run_name="c10q1"
  $(echo $EXPERIMENT) --num_neg_samples=1 --context_size=10 --run_name="c10nns1"
  $(echo $EXPERIMENT) --num_neg_samples=2 --context_size=10 --run_name="c10nns2"
  $(echo $EXPERIMENT) --num_neg_samples=5 --context_size=10 --run_name="c10nns5"
done

### Tune Embedding
#for i in `seq 1 3`; do
#  EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_feat2 $ARGS_BASE $EMBED_DS --lr=0.02"
#  $(echo $EXPERIMENT) --run_name="feat" $(echo $ARGS_DS)
#  $(echo $EXPERIMENT) --run_name="embed" $(echo "$ARGS_DS $EMBED_DS")
#  $(echo $EXPERIMENT) --run_name="embed_full" $(echo "$ARGS_DS $EMBED_FULL_DS")
#done
