#!/bin/bash
# Runs the compiled parallel-transpose unit test under mpirun on several
# rank counts. The test builds a distributed block-tridiagonal matrix,
# assembles the consistent parallel transpose, and checks every owned row
# against the serial transpose (see test_parallelTranspose.cpp). Skips
# itself when no MPI launcher is available.
#
# Usage: run-parallel-transpose-test.sh <test_parallelTranspose exe> [nprocs...]
set -u

EXE=${1:?usage: $0 <test_parallelTranspose exe> [nprocs...]}
shift
NPROCS=${*:-2 3 4}

MPIRUN=${MPIRUN:-mpirun}
if ! command -v "${MPIRUN}" >/dev/null 2>&1; then
    echo "SKIP: ${MPIRUN} not found (no MPI launcher)"
    exit 0
fi

status=0
for np in ${NPROCS}; do
    echo "=== np=${np} ==="
    "${MPIRUN}" -np "${np}" --oversubscribe "${EXE}" || status=1
done
exit ${status}
