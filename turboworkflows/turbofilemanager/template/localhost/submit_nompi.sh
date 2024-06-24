#!/bin/bash

# Note:
# The variables _xxx_ are replaced by job_manager.py. The pre-defined variables are
# _INPUT_, _OUTPUT_, _PREOPTION_, _POSTOPTION_, _JOBNAME_, _BINARY_ROOT_, and _BINARY_.
# Others defined in queue_data.toml (e.g. _NUM_CORE_) are also replaced by job_manager.py 
# so that one can manually define other variables needed for submitting jobs to a queueing 
# system.

export OMP_NUM_THREADS=_OMP_NUM_THREADS_

CORES=_NUM_CORES_
INPUT=_INPUT_
OUTPUT=_OUTPUT_
PREOPTION=_PREOPTION_
POSTOPTION=_POSTOPTION_
BINARY=_BINARY_ROOT_/_BINARY_

$BINARY $PREOPTION < $INPUT $POSTOPTION > $OUTPUT
