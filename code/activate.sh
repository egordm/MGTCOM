#!/bin/bash

BASEPATH=$(realpath $(dirname $BASH_SOURCE))
if [ ! $CONDA_PREFIX ]; then conda activate ./env; fi;
export PYTHONPATH="$PYTHONPATH:$BASEPATH/shared:$BASEPATH/datasets"