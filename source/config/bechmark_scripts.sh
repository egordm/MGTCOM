# Reprdim Tuning
EXPERIMENT="python ml/ml/executors/mgcom_topo_executor.py --embedding_visualizer.dim_reduction_mode=TSNE --experiment='mgtcom_tune_reprdim' --project='ThesisExp'"
$(echo $EXPERIMENT) --repr_dim=64 --run_name="d64"  --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32" --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=16 --run_name="d16" --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=8  --run_name="d8"  --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=4  --run_name="d4"  --batch_size=128 --max_epochs=40 --dataset=Cora

# Temporal Baseline CTDNE
EXPERIMENT="python ml/ml/executors/baselines/ctdne_executor.py --embedding_visualizer.dim_reduction_mode=TSNE --experiment='ctdne_baseline' --project='ThesisExp'"
$(echo $EXPERIMENT) --repr_dim=64 --run_name="d64"  --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32" --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=16 --run_name="d16" --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=8  --run_name="d8"  --batch_size=128 --max_epochs=40 --dataset=Cora
$(echo $EXPERIMENT) --repr_dim=4  --run_name="d4"  --batch_size=128 --max_epochs=40 --dataset=Cora