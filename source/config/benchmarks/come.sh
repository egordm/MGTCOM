# Export Datasets
#EXPORT_SCRIPT="python datasets/datasets/cli/export_dataset.py --version=base"
#$(echo $EXPORT_SCRIPT) --dataset=DBLPHCN
#$(echo $EXPORT_SCRIPT) --dataset=HouseOfRepresentativesCongress116
#$(echo $EXPORT_SCRIPT) --dataset=ICEWS0515
#$(echo $EXPORT_SCRIPT) --dataset=IMDB5000
#$(echo $EXPORT_SCRIPT) --dataset=SocialDistancingStudents
#$(echo $EXPORT_SCRIPT) --dataset=StarWars
#$(echo $EXPORT_SCRIPT) --dataset=Cora

# Base args
ARGS_BASE="--project=MGTCOM"
REPR_DIM=64

# Benchmark ComE
run_ComE () {
  PREFIX=$(echo $RANDOM | md5sum | head -c 6; echo)
  RUN_NAME="R${PREFIX}_r$2_k$3"
  conda activate ./benchmarks/ComE/env
  python benchmarks/ComE/main.py --batch_size=256 --representation_size=$2 --dataset=$1 --k=$3 --run_name=$RUN_NAME
  conda deactivate
  source activate.sh
  python ml/ml/executors/evaluate.py $(echo "$ARGS_BASE") --model=ComE --run_name=$RUN_NAME --dataset=$1 --k=$3 --repr_dim=$2 --experiment=baseline_ComE64
  conda deactivate
}

#run_ComE DBLPHCN $REPR_DIM 10
run_ComE DBLPHCN $REPR_DIM 20
#run_ComE HouseOfRepresentativesCongress116 $REPR_DIM 10
#run_ComE HouseOfRepresentativesCongress116 $REPR_DIM 20
#run_ComE ICEWS0515 $REPR_DIM 10
run_ComE ICEWS0515 $REPR_DIM 20
#run_ComE IMDB5000 $REPR_DIM 10
run_ComE IMDB5000 $REPR_DIM 20
#run_ComE SocialDistancingStudents $REPR_DIM 10
run_ComE SocialDistancingStudents $REPR_DIM 20
#run_ComE StarWars $REPR_DIM 10
run_ComE StarWars $REPR_DIM 20
#run_ComE StarWars $REPR_DIM 7
#run_ComE Cora $REPR_DIM 10
run_ComE Cora $REPR_DIM 20
