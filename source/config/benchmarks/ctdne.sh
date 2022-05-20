source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_DBLP="--batch_size=128 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_IMDB="--batch_size=128 --max_epochs=50 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_ICEWS="--batch_size=128 --max_epochs=50 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_SDS="--batch_size=256 --max_epochs=20 --dataset=SocialDistancingStudents --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"

ARGS_DS="$ARGS_DBLP"

EXPERIMENT="$ARGS_CMD/baselines/ctdne_executor.py --experiment=baseline_ctdne $ARGS_BASE $ARGS_DS --num_workers=3 --metric=DOTP --infer_k=20 --loss=SKIPGRAM --lr=0.04 --cpu"

ARGS_DS="$ARGS_DBLP"
$(echo $EXPERIMENT) --run_name="ctdne" $(echo $ARGS_DS)

ARGS_DS="$ARGS_IMDB"
$(echo $EXPERIMENT) --run_name="ctdne" $(echo $ARGS_DS)

ARGS_DS="$ARGS_ICEWS"
$(echo $EXPERIMENT) --run_name="ctdne" $(echo $ARGS_DS)

ARGS_DS="$ARGS_SDS"
$(echo $EXPERIMENT) --run_name="ctdne" $(echo $ARGS_DS)
