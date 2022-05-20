

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
