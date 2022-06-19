ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM --lr=0.02 --num_workers=3 --metric=DOTP"
ARGS_SW="--batch_size=16  --max_epochs=200 --dataset=StarWars --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_CORA="--batch_size=128 --max_epochs=50 --dataset=Cora --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_DBLP="--batch_size=128 --max_epochs=50 --dataset=DBLPHCN --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_IMDB="--batch_size=128 --max_epochs=50 --dataset=IMDB5000 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_ICEWS="--batch_size=128 --max_epochs=50 --dataset=ICEWS0515 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"
ARGS_SDS="--batch_size=2048 --max_epochs=40 --dataset=SocialDistancingStudents --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=10 --classification_eval.interval=10"

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

for i in `seq 1 3`; do
  # Tune Ballroom Walk
  EXPERIMENT="$ARGS_CMD/mgcom_tempo_executor.py --experiment=mgtcom_tune_ballroom3 $ARGS_BASE $ARGS_DS"
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


# Tune E2E
#EXPERIMENT="$ARGS_CMD/mgcom_e2e_executor.py --experiment=mgtcom_e2e $ARGS_BASE $ARGS_DS --classification_eval.interval=10"
#$(echo $EXPERIMENT) --batch_size=200 --max_epochs=200 --prior_sigma_scale=0.05 --use_tempo=false --offline

