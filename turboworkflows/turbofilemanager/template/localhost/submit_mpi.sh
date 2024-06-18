#!/bin/bash
#SBATCH --account=_QUEUE_
#SBATCH --partition=_QUEUE_
#SBATCH --job-name=_JOBNAME_
#SBATCH --nodes=_NODES_
#SBATCH --ntasks=_NUM_CORES_
#SBATCH --time=_MAX_TIME_
#SBATCH --ntasks-per-node=_MPI_PER_NODE_

# Note:
# The variables _xxx_ are replaced by job_manager.py.

export OMP_NUM_THREADS=_OMP_NUM_THREADS_

CORES=_NUM_CORES_
INPUT=_INPUT_
OUTPUT=_OUTPUT_
PREOPTION=_PREOPTION_
POSTOPTION=_POSTOPTION_
BINARY=_BINARY_ROOT_/_BINARY_

srun -np $CORES $BINARY $PREOPTION < $INPUT $POSTOPTION > $OUTPUT

