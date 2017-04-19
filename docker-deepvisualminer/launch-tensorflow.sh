#!/bin/bash

# Script to launch a build container for building Tensorflow from sources.
# This is a not a mandatory step. The idea here is to build a custom Tensorflow
# with all optimizations and instruction sets enabled, unlike stock Tensorflow 
# python package which does not have any of those enabled.

mkdir -p ./shared

echo "Launching Tensorflow build container"
sudo docker run  --rm --name tensorflowbuild \
    -v $(pwd)/shared:/mnt/shared \
    -v $(pwd)/build-tensorflow.sh:/root/build-tensorflow.sh \
    ubuntu:16.04 \
    /root/build-tensorflow.sh

sudo chown -R $(logname):$(logname) ./shared
