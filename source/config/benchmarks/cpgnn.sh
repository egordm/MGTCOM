# Export Datasets
#EXPORT_SCRIPT="python datasets/datasets/cli/export_dataset.py --version=cpgnn"
#$(echo $EXPORT_SCRIPT) --dataset=DBLPHCN
#$(echo $EXPORT_SCRIPT) --dataset=HouseOfRepresentativesCongress116
#$(echo $EXPORT_SCRIPT) --dataset=ICEWS0515
#$(echo $EXPORT_SCRIPT) --dataset=IMDB5000
#$(echo $EXPORT_SCRIPT) --dataset=SocialDistancingStudents
#$(echo $EXPORT_SCRIPT) --dataset=StarWars
#$(echo $EXPORT_SCRIPT) --dataset=Cora

ARGS_BASE="--project=MGTCOM"

# Benchmark GEMSEC
run_GEMSEC() {
  PREFIX=$(echo $RANDOM | md5sum | head -c 6; echo)
  RUN_NAME="R${PREFIX}_r$2_k$3"
  conda activate ./benchmarks/CP-GNN/env
  python benchmarks/CP-GNN/main.py --dataset=$1 --dataset_version=cpgnn --run_name=$RUN_NAME --repr_dim=$2 --primary_type=$3
  conda deactivate
  source activate.sh
  python ml/ml/executors/evaluate.py $(echo "$ARGS_BASE") --model=CP-GNN --run_name=$RUN_NAME --dataset=$1 --dataset_version=cpgnn --k=20 --repr_dim=$2 --experiment=baseline_CPGNN
  conda deactivate
}

REPR_DIM=64

run_GEMSEC DBLPHCN $REPR_DIM Paper
run_GEMSEC IMDB5000 $REPR_DIM Movie
run_GEMSEC ICEWS0515 $REPR_DIM Entity
run_GEMSEC StarWars $REPR_DIM Character
run_GEMSEC Cora $REPR_DIM 0
#run_GEMSEC SocialDistancingStudents $REPR_DIM Tweet


