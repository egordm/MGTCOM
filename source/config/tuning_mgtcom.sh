ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_DBLP="--batch_size=128 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
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
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_reprdim $ARGS_BASE $ARGS_DS"
#$(echo $EXPERIMENT) --repr_dim=64 --run_name="d64"
#$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32"
#$(echo $EXPERIMENT) --repr_dim=16 --run_name="d16"
#$(echo $EXPERIMENT) --repr_dim=8  --run_name="d8"
#$(echo $EXPERIMENT) --repr_dim=4  --run_name="d4"
#$(echo $EXPERIMENT) --repr_dim=16 --conv_hidden_dim=32 --run_name="d16h32"
#$(echo $EXPERIMENT) --repr_dim=8  --conv_hidden_dim=16 --run_name="d8h32"
#$(echo $EXPERIMENT) --repr_dim=4  --conv_hidden_dim=16 --run_name="d4h32"
#
## Tune LR
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_lr $ARGS_BASE $ARGS_DS"
#$(echo $EXPERIMENT) --lr=0.1    --run_name="1e-1"
#$(echo $EXPERIMENT) --lr=0.01   --run_name="1e-2"
#$(echo $EXPERIMENT) --lr=0.02   --run_name="2e-2"
#$(echo $EXPERIMENT) --lr=0.05   --run_name="5e-2"
#$(echo $EXPERIMENT) --lr=0.001  --run_name="1e-3"
#$(echo $EXPERIMENT) --lr=0.0001 --run_name="1e-4"
#
## Tune Hinge Margin
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_hmargin $ARGS_BASE $ARGS_DS"
#$(echo $EXPERIMENT) --hinge_margin=0.5 --run_name="hmargin=0.5"
#$(echo $EXPERIMENT) --hinge_margin=1   --run_name="hmargin=1"
#$(echo $EXPERIMENT) --hinge_margin=2   --run_name="hmargin=2"
#$(echo $EXPERIMENT) --hinge_margin=4   --run_name="hmargin=4"
#$(echo $EXPERIMENT) --hinge_margin=8   --run_name="hmargin=8"
#$(echo $EXPERIMENT) --hinge_margin=16  --run_name="hmargin=16"
#
## Tune Neighbor Sampling
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_neighbors $ARGS_BASE $ARGS_DS"
#$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 3 2 --run_name="hgt_b32"
#$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 3 1 --run_name="hgt_b31"
#$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 2 2 --run_name="hgt_b22"
#$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 6 2 --run_name="hgt_b62"
#$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 8 4 --run_name="hgt_b84"
#$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 3 2 --run_name="sage_b32"
#$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 3 1 --run_name="sage_b31"
#$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 2 2 --run_name="sage_b22"
#$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 6 2 --run_name="sage_b62"
#$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 8 4 --run_name="sage_b84"
#
## Tune Random Walk
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_rw $ARGS_BASE $ARGS_DS"
#$(echo $EXPERIMENT) --context_size=20 --run_name="c20"
#$(echo $EXPERIMENT) --context_size=10 --run_name="c10"
#$(echo $EXPERIMENT) --context_size=5  --run_name="c5"
#$(echo $EXPERIMENT) --context_size=3  --run_name="c3"
#$(echo $EXPERIMENT) --walks_per_node=10 --context_size=10 --run_name="c10w10"
#$(echo $EXPERIMENT) --walks_per_node=20 --context_size=10 --run_name="c10w20"
#$(echo $EXPERIMENT) --walks_per_node=5  --context_size=10 --run_name="c10w5"
#$(echo $EXPERIMENT) --q=0.5  --context_size=10 --run_name="c10q0.5"
#$(echo $EXPERIMENT) --q=0.25 --context_size=10 --run_name="c10q0.25"
#$(echo $EXPERIMENT) --q=0.1  --context_size=10 --run_name="c10q0.1"
#$(echo $EXPERIMENT) --q=1    --context_size=10 --run_name="c10q1"
#
## Tune Embedding
#EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_feat $ARGS_BASE"
#$(echo $EXPERIMENT) --run_name="feat" $(echo $ARGS_DS)
#$(echo $EXPERIMENT) --run_name="embed" $(echo $EMBED_DS)
#$(echo $EXPERIMENT) --run_name="embed_full" $(echo $EMBED_FULL_DS)

for i in `seq 1 3`; do
  # Tune Ballroom Walk
  EXPERIMENT="$ARGS_CMD/mgcom_tempo_executor.py --experiment=mgtcom_tune_ballroom $ARGS_BASE $ARGS_DS"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --context_size=20 --run_name="7c20"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --context_size=10 --run_name="7c10"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --context_size=5  --run_name="7c5"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --context_size=3  --run_name="7c3"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walks_per_node=30 --context_size=10 --run_name="7c10w30"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walks_per_node=20 --context_size=10 --run_name="7c10w20"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walks_per_node=15 --context_size=10 --run_name="7c10w15"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walks_per_node=10 --context_size=10 --run_name="7c10w10"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walks_per_node=5  --context_size=10 --run_name="7c10w5"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --num_neg_samples=1 --context_size=10 --run_name="7c10nns1"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --num_neg_samples=2 --context_size=10 --run_name="7c10nns2"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --num_neg_samples=5 --context_size=10 --run_name="7c10nns2"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walk_length=80 --context_size=10 --run_name="7c10wl80"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walk_length=40 --context_size=10 --run_name="7c10wl40"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walk_length=20 --context_size=10 --run_name="7c10wl20"
  $(echo $EXPERIMENT) $(echo $EMBED_DS) --walk_length=10 --context_size=10 --run_name="7c10wl10"
done


## Tune E2E
#EXPERIMENT="$ARGS_CMD/mgcom_e2e_executor.py --experiment=mgtcom_e2e $ARGS_BASE $ARGS_DS --classification_eval.interval=10"
#$(echo $EXPERIMENT) --batch_size=200 --max_epochs=200 --prior_sigma_scale=0.05 --use_tempo=false --offline