# Export Datasets
EXPORT_SCRIPT="python datasets/datasets/cli/export_dataset.py --version=base"
$(echo $EXPORT_SCRIPT) --dataset=DBLPHCN
$(echo $EXPORT_SCRIPT) --dataset=HouseOfRepresentativesCongress116
$(echo $EXPORT_SCRIPT) --dataset=ICEWS0515
$(echo $EXPORT_SCRIPT) --dataset=IMDB5000
$(echo $EXPORT_SCRIPT) --dataset=SocialDistancingStudents
$(echo $EXPORT_SCRIPT) --dataset=StarWars

ARGS_BASE="--project=MGTCOM"

# Benchmark ComE
run_ComE () {
  PREFIX=$(echo $RANDOM | md5sum | head -c 6; echo)
  RUN_NAME="R${PREFIX}_r$2_k$3"
  conda activate /data/pella/projects/University/Thesis/Thesis/source/benchmarks/ComE/env
  python benchmarks/ComE/main.py --batch_size=128 --representation_size=$2 --dataset=$1 --k=$3 --run_name=$RUN_NAME
  conda deactivate
  source activate.sh
  python ml/ml/executors/evaluate.py $(echo "$ARGS_BASE") --model=ComE --run_name=$RUN_NAME --dataset=$1 --k=$3 --repr_dim=$2 --experiment=baseline_ComE
  conda deactivate
}

run_ComE DBLPHCN 32 10
run_ComE DBLPHCN 32 20
run_ComE HouseOfRepresentativesCongress116 32 10
run_ComE HouseOfRepresentativesCongress116 32 20
run_ComE ICEWS0515 32 10
run_ComE ICEWS0515 32 20
run_ComE IMDB5000 32 10
run_ComE IMDB5000 32 20
run_ComE SocialDistancingStudents 32 10
run_ComE SocialDistancingStudents 32 20
run_ComE StarWars 32 10
run_ComE StarWars 32 20
run_ComE StarWars 32 7
run_ComE Cora 32 10
run_ComE Cora 32 20

# Benchmark GEMSEC
ARGS_BASE="--project=MGTCOM"
run_GEMSEC() {
  PREFIX=$(echo $RANDOM | md5sum | head -c 6; echo)
  RUN_NAME="R${PREFIX}_r$2_k$3"
  conda activate /data/pella/projects/University/Thesis/Thesis/source/benchmarks/GEMSEC/env
  CUDA_VISIBLE_DEVICES="" python benchmarks/GEMSEC/src/embedding_clustering.py --dataset=$1 --cluster-number=$3 --run_name=$RUN_NAME --num-of-walks=10 --dimensions=$2
  conda deactivate
  source activate.sh
  python ml/ml/executors/evaluate.py $(echo "$ARGS_BASE") --model=GEMSEC --run_name=$RUN_NAME --dataset=$1 --k=$3 --repr_dim=$2 --experiment=baseline_GEMSEC
  conda deactivate
}

run_GEMSEC DBLPHCN 32 10
run_GEMSEC DBLPHCN 32 20
run_GEMSEC HouseOfRepresentativesCongress116 32 10
run_GEMSEC HouseOfRepresentativesCongress116 32 20
run_GEMSEC ICEWS0515 32 10
run_GEMSEC ICEWS0515 32 20
run_GEMSEC IMDB5000 32 10
run_GEMSEC IMDB5000 32 20
run_GEMSEC SocialDistancingStudents 32 10
run_GEMSEC SocialDistancingStudents 32 20
run_GEMSEC StarWars 32 10
run_GEMSEC StarWars 32 20
run_GEMSEC StarWars 32 7
run_GEMSEC Cora 32 10
run_GEMSEC Cora 32 20


