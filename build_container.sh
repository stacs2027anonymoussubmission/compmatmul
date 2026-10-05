#!/bin/bash

# This is an example script for building the project in an apptainer container
# Adjust the architecture and build options as appropriate

if [ $(uname -m) = "x86_64" ]
then
    ARCH=emeraldrapids
    TUNE=emeraldrapids

    FILENAME=compmatmul_mkl_mkl_ffht.sif
    if [ ! -e ${FILENAME} ]
    then
        apptainer build --build-arg BLAS_BACKEND=MKL --build-arg FFT_BACKEND=MKL \
            --build-arg FWHT_BACKEND=FFHT --build-arg ARCH=${ARCH} \
            --build-arg TUNE=${TUNE} ${FILENAME} Apptainer.def
    fi
fi
