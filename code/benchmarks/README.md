# Benchmarks
See `config/benchmarks/*.yml` for configuration

#### Running a single benchmark
```bash
benchmarks/benchmarks/scripts/execute.py ANGEL ucidata-zachary static
```

#### Evaluating a single benchmark
```bash
benchmarks/benchmarks/scripts/evaluate.py GreeneDCD-louvain DBLP-HCN split_5 2022-01-24_16-13-13-GreeneDCD-louvain-DBLP-HCN:split_5
```

#### Evaluating and running a single benchmark
```bash
benchmarks/benchmarks/scripts/execute_and_evaluate.py ANGEL ucidata-zachary static
```

#### Tuning a benchmark
```bash
benchmarks/benchmarks/scripts/tune.py ESPRA star-wars
```