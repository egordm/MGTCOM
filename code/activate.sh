#!/bin/bash

BASEPATH=$(realpath $(dirname ${BASH_SOURCE:-$_}))
conda activate $BASEPATH/env;
export PYTHONPATH="$PYTHONPATH:$BASEPATH/shared:$BASEPATH/datasets:$BASEPATH/ml"
