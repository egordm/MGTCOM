ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_DBLP="--batch_size=512 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_IMDB="--batch_size=128 --max_epochs=50 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"
ARGS_ICEWS="--batch_size=128 --max_epochs=50 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --classification_eval.interval=4"


# Tune Repr dim
EXPERIMENT="$ARGS_CMD/baselines/cpgnn_executor.py --experiment=baseline_CPGNN $ARGS_BASE $ARGS_DS"
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32" --k=10 --max_epochs=200 --metric=DOTP
$(echo $EXPERIMENT) --repr_dim=32 --run_name="d32" --k=20 --max_epochs=200 --metric=DOTP