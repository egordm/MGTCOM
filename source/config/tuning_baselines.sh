source activate.sh
ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"

# Tune CP-GNN
ARGS_DBLP="--batch_size=512 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"

ARGS_DS="$ARGS_DBLP"

EXPERIMENT="$ARGS_CMD/baselines/cpgnn_executor.py --experiment=tuning_CPGNN $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32mDOTP" --k=10 --max_epochs=100 --metric=DOTP
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32mL2"   --k=10 --max_epochs=100 --metric=L2
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32a1"   --k=10 --max_epochs=100 --metric=DOTP --num_layers_aux=1 --num_samples 3 2 2 2
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32a1d"   --k=10 --max_epochs=100 --metric=DOTP --num_layers_aux=1 --num_samples 6 4 4 4
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32a2"   --k=10 --max_epochs=100 --metric=DOTP --num_layers_aux=2 --num_samples 3 2 2 2 2
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32a2d"   --k=10 --max_epochs=100 --metric=DOTP --num_layers_aux=2 --num_samples 6 4 4 4 4
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32kl=3"   --k=10 --max_epochs=100 --metric=DOTP --k_length=3 --num_samples 3 2 2 2
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32kl=3d"   --k=10 --max_epochs=100 --metric=DOTP --k_length=3 --num_samples 6 4 4 4
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32kl=4"   --k=10 --max_epochs=100 --metric=DOTP --k_length=4 --num_samples 3 2 2 2 2
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32kl=4d"   --k=10 --max_epochs=100 --metric=DOTP --k_length=4 --num_samples 6 4 4 4 4
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32kl=5"   --k=10 --max_epochs=100 --metric=DOTP --k_length=5 --num_samples 3 2 2 2 2 2
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32kl=5d"   --k=10 --max_epochs=100 --metric=DOTP --k_length=5 --num_samples 6 4 4 4 4 4
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32mDOTP" --k=10 --max_epochs=100 --metric=DOTP

# Tune CP-GNN
EXPERIMENT="$ARGS_CMD/baselines/cpgnn_executor.py --experiment=tuning_CPGNN $ARGS_BASE $ARGS_DS"

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