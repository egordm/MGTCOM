source activate.sh

ARGS_CMD="python ml/ml/executors"
ARGS_BASE="--project=MGTCOM2 --embedding_visualizer.dim_reduction_mode=TSNE --embedding_visualizer.interval=100 --classification_eval.interval=100 --clustering_visualizer.interval=100"

EXPERIMENT="$ARGS_CMD/mgcom_comdet_executor.py --dataset=DBLPHCN --experiment=abl_clustering $ARGS_BASE --max_epochs=1000 --batch_size=400 \
  --pretrained_path=/data/pella/projects/University/Thesis/Thesis/source/config/ablations/dblp_embeddings_hetero.pt --cpu"

#for i in `seq 1 3`; do
#  $(echo $EXPERIMENT) --run_name="s=1"    --init_k=3 --prior_sigma_scale=1    --prior_alpha=10 --prior_nu=65 --prior_kappa=1
#  $(echo $EXPERIMENT) --run_name="s=0.85"    --init_k=3 --prior_sigma_scale=0.85    --prior_alpha=10 --prior_nu=65 --prior_kappa=1
#  $(echo $EXPERIMENT) --run_name="s=0.75"    --init_k=3 --prior_sigma_scale=0.75    --prior_alpha=10 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="s=0.5"    --init_k=3 --prior_sigma_scale=0.5    --prior_alpha=10 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="s=0.05"   --init_k=3 --prior_sigma_scale=0.05   --prior_alpha=10 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="s=0.005"  --init_k=3 --prior_sigma_scale=0.005  --prior_alpha=10 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="s=0.0005" --init_k=3 --prior_sigma_scale=0.0005 --prior_alpha=10 --prior_nu=65 --prior_kappa=1
#done

#for i in `seq 1 3`; do
##  $(echo $EXPERIMENT) --run_name="a=1"    --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=1    --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="a=10"   --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="a=100"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=100  --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="a=1000" --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=1000 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="a=10000" --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10000 --prior_nu=65 --prior_kappa=1
#  $(echo $EXPERIMENT) --run_name="a=50000" --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=50000 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="a=100000" --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=100000 --prior_nu=65 --prior_kappa=1
#  $(echo $EXPERIMENT) --run_name="a=500000" --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=500000 --prior_nu=65 --prior_kappa=1
##  $(echo $EXPERIMENT) --run_name="a=1000000" --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=1000000 --prior_nu=65 --prior_kappa=1
#done


for i in `seq 1 3`; do
#    $(echo $EXPERIMENT) --run_name="k=0.001"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=0.001
    $(echo $EXPERIMENT) --run_name="k=0.005"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=0.005
#    $(echo $EXPERIMENT) --run_name="k=0.01"   --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=0.01
    $(echo $EXPERIMENT) --run_name="k=0.05"   --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=0.05
#    $(echo $EXPERIMENT) --run_name="k=0.1"    --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=0.1
#    $(echo $EXPERIMENT) --run_name="k=1"      --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=1
#    $(echo $EXPERIMENT) --run_name="k=10"     --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=10
    $(echo $EXPERIMENT) --run_name="k=50"     --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=50
#    $(echo $EXPERIMENT) --run_name="k=100"    --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=100
    $(echo $EXPERIMENT) --run_name="k=500"    --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=500
#    $(echo $EXPERIMENT) --run_name="k=1000"   --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10   --prior_nu=65 --prior_kappa=1000
done


for i in `seq 1 3`; do
    $(echo $EXPERIMENT) --run_name="nu=65"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=65 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=70"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=70 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=80"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=80 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=90"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=90 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=100"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=100 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=120"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=120 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=140"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=140 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=180"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=180 --prior_kappa=1
    $(echo $EXPERIMENT) --run_name="nu=200"  --init_k=3 --prior_sigma_scale=0.5 --prior_alpha=10 --prior_nu=200 --prior_kappa=1
done