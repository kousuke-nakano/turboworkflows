#!/bin/bash
# Note:
# The variables inside _xxx_ are replaced by job_manager.py.
# Implemented arguments are:
# QUEUE, JOBNAME, MAX_TIME, NODES, CORES_PER_NODE, MPI_PER_NODE
# OMP_NUM_THREADS, NUM_CORES, INPUT, OUTPUT, BINARY_ROOT, BINARY

export OMP_NUM_THREADS=1

CORES=1
INPUT=convertfort10mol.input
OUTPUT=out_mol
PREOPTION=_PREOPTION_
POSTOPTION=_POSTOPTION_
BINARY=convertfort10mol.x

$BINARY  < $INPUT  > $OUTPUT
