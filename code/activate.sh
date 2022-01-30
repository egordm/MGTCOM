#!/bin/bash

BASEPATH=$(realpath $(dirname ${BASH_SOURCE:-$0}))
conda activate ./env;
export PYTHONPATH="$PYTHONPATH:$BASEPATH/shared:$BASEPATH/datasets:$BASEPATH/benchmarks"