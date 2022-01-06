#!/bin/bash

BASEPATH=$(realpath $(dirname ${BASH_SOURCE:-$0}))
if [ ! $CONDA_PREFIX ]; then conda activate ./env; fi;
export PYTHONPATH="$PYTHONPATH:$BASEPATH/shared:$BASEPATH/datasets"