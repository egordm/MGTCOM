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

# Tune Repr dim
EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_reprdim $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --repr_dim=64 --run_name="d64"
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32"
$(echo $EXPERIMENT) --repr_dim=16 --run_name="d16"
$(echo $EXPERIMENT) --repr_dim=8  --run_name="d8"
$(echo $EXPERIMENT) --repr_dim=4  --run_name="d4"
$(echo $EXPERIMENT) --repr_dim=16 --conv_hidden_dim=32 --run_name="d16h32"
$(echo $EXPERIMENT) --repr_dim=8  --conv_hidden_dim=16 --run_name="d8h32"
$(echo $EXPERIMENT) --repr_dim=4  --conv_hidden_dim=16 --run_name="d4h32"

# Tune LR
EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_lr $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --lr=0.1    --run_name="1e-1"
$(echo $EXPERIMENT) --lr=0.01   --run_name="1e-2"
$(echo $EXPERIMENT) --lr=0.02   --run_name="2e-2"
$(echo $EXPERIMENT) --lr=0.05   --run_name="5e-2"
$(echo $EXPERIMENT) --lr=0.001  --run_name="1e-3"
$(echo $EXPERIMENT) --lr=0.0001 --run_name="1e-4"

# Tune Hinge Margin
EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_hmargin $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --hinge_margin=0.5 --run_name="hmargin=0.5"
$(echo $EXPERIMENT) --hinge_margin=1   --run_name="hmargin=1"
$(echo $EXPERIMENT) --hinge_margin=2   --run_name="hmargin=2"
$(echo $EXPERIMENT) --hinge_margin=4   --run_name="hmargin=4"
$(echo $EXPERIMENT) --hinge_margin=8   --run_name="hmargin=8"
$(echo $EXPERIMENT) --hinge_margin=16  --run_name="hmargin=16"

# Tune Neighbor Sampling
EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_neighbors $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 3 2 --run_name="hgt_b32"
$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 3 1 --run_name="hgt_b31"
$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 2 2 --run_name="hgt_b22"
$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 6 2 --run_name="hgt_b62"
$(echo $EXPERIMENT) --conv_method=HGT  --num_samples 8 4 --run_name="hgt_b84"
$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 3 2 --run_name="sage_b32"
$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 3 1 --run_name="sage_b31"
$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 2 2 --run_name="sage_b22"
$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 6 2 --run_name="sage_b62"
$(echo $EXPERIMENT) --conv_method=SAGE --num_samples 8 4 --run_name="sage_b84"

# Tune Random Walk
EXPERIMENT="$ARGS_CMD/mgcom_topo_executor.py --experiment=mgtcom_tune_rw $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --context_size=20 --run_name="c20"
$(echo $EXPERIMENT) --context_size=10 --run_name="c10"
$(echo $EXPERIMENT) --context_size=5  --run_name="c5"
$(echo $EXPERIMENT) --context_size=3  --run_name="c3"
$(echo $EXPERIMENT) --walks_per_node=10 --context_size=10 --run_name="c10w10"
$(echo $EXPERIMENT) --walks_per_node=20 --context_size=10 --run_name="c10w20"
$(echo $EXPERIMENT) --walks_per_node=5  --context_size=10 --run_name="c10w5"
$(echo $EXPERIMENT) --q=0.5  --context_size=10 --run_name="c10q0.5"
$(echo $EXPERIMENT) --q=0.25 --context_size=10 --run_name="c10q0.25"
$(echo $EXPERIMENT) --q=0.1  --context_size=10 --run_name="c10q0.1"
$(echo $EXPERIMENT) --q=1    --context_size=10 --run_name="c10q1"

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

# Tune CTDNE Repr Dim
EXPERIMENT="$ARGS_CMD/baselines/ctdne_executor.py --experiment=ctdne_tune_repr $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --repr_dim=64 --run_name="d64"
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32"
$(echo $EXPERIMENT) --repr_dim=16 --run_name="d16"
$(echo $EXPERIMENT) --repr_dim=8  --run_name="d8"
$(echo $EXPERIMENT) --repr_dim=4  --run_name="d4"

# Tune CTDNE Walk
EXPERIMENT="$ARGS_CMD/baselines/ctdne_executor.py --experiment=ctdne_tune_walk $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --context_size=20 --run_name="c20"
$(echo $EXPERIMENT) --context_size=10 --run_name="c10"
$(echo $EXPERIMENT) --context_size=5  --run_name="c5"
$(echo $EXPERIMENT) --context_size=3  --run_name="c3"
$(echo $EXPERIMENT) --walks_per_node=10 --context_size=10 --run_name="c10w10"
$(echo $EXPERIMENT) --walks_per_node=20 --context_size=10 --run_name="c10w20"
$(echo $EXPERIMENT) --walks_per_node=5  --context_size=10 --run_name="c10w5"
$(echo $EXPERIMENT) --q=0.5  --context_size=10 --run_name="c10q0.5"
$(echo $EXPERIMENT) --q=0.25 --context_size=10 --run_name="c10q0.25"
$(echo $EXPERIMENT) --q=0.1  --context_size=10 --run_name="c10q0.1"
$(echo $EXPERIMENT) --q=1    --context_size=10 --run_name="c10q1"

# Benchmark Topo Link Prediction + Node Classification
python ml/ml/executors/baselines/het2vec_executor.py $(echo "$ARGS_BASE $ARGS_DS") --experiment=bench_topo --run_name="node2vec"
python ml/ml/executors/mgcom_topo_executor.py        $(echo "$ARGS_BASE $ARGS_DS") --experiment=bench_topo --run_name="mgtcom"
python ml/ml/executors/mgcom_topo_executor.py        $(echo "$ARGS_BASE $ARGS_DS") --experiment=bench_topo --run_name="mgtcom_embed" $(echo $EMBED_DS)
python ml/ml/executors/mgcom_topo_executor.py        $(echo "$ARGS_BASE $ARGS_DS") --experiment=bench_topo --run_name="mgtcom_embedf" $(echo $EMBED_FULL_DS)

# Benchmark Topo Inference
ARGS_INF="--eval_inference=true --split_force=true --split_num_val=0.1"
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.1 --run_name="80_HGT"  --conv_method=HGT
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.1 --run_name="80_SAGE" --conv_method=SAGE
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.1 --run_name="80_mgtcom_embed" $(echo $EMBED_DS)
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.1 --run_name="80_mgtcom_embedf" $(echo $EMBED_FULL_DS)

python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.3 --run_name="60_HGT"  --conv_method=HGT
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.3 --run_name="60_SAGE" --conv_method=SAGE
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.3 --run_name="60_mgtcom_embed" $(echo $EMBED_DS)
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.3 --run_name="60_mgtcom_embedf" $(echo $EMBED_FULL_DS)

python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.5 --run_name="40_HGT"  --conv_method=HGT
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.5 --run_name="40_SAGE" --conv_method=SAGE
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.5 --run_name="40_mgtcom_embed" $(echo $EMBED_DS)
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.5 --run_name="40_mgtcom_embedf" $(echo $EMBED_FULL_DS)

python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.7 --run_name="20_HGT"  --conv_method=HGT
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.7 --run_name="20_SAGE" --conv_method=SAGE
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.7 --run_name="20_mgtcom_embed" $(echo $EMBED_DS)
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.7 --run_name="20_mgtcom_embedf" $(echo $EMBED_FULL_DS)

python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.8 --run_name="10_HGT"  --conv_method=HGT
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.8 --run_name="10_SAGE" --conv_method=SAGE
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.8 --run_name="10_mgtcom_embed" $(echo $EMBED_DS)
python ml/ml/executors/mgcom_topo_executor.py $(echo "$ARGS_BASE $ARGS_DS $ARGS_INF") --eval_inference --experiment=bench_inference --split_num_test=0.8 --run_name="10_mgtcom_embedf" $(echo $EMBED_FULL_DS)
