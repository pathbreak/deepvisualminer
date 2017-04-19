#!/bin/bash

# Build script for building TensorFlow from sources with all advanced 
# instruction sets enabled - SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA.
#
# Code based on 
#   - https://www.tensorflow.org/install/install_sources 
#   - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel

apt-get update 

apt-get install -y --no-install-recommends build-essential curl git libcurl3-dev \
    libfreetype6-dev libpng12-dev libzmq3-dev pkg-config rsync \
    python python3 python3-numpy python3-dev python3-pip python3-wheel python3-setuptools \
    openjdk-8-jdk software-properties-common  unzip zip zlib1g-dev 

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
apt-get update
apt-get install -y bazel

cd ~

git clone https://github.com/tensorflow/tensorflow 
cd tensorflow
git fetch
git pull
git checkout r1.0

export TF_ENABLE_XLA=0     
export TF_NEED_CUDA=0      
export TF_NEED_GCP=0       
export TF_NEED_HDFS=0      
export TF_NEED_JEMALLOC=1  
export TF_NEED_OPENCL=0
export PYTHON_BIN_PATH=/usr/bin/python3
export PYTHON_LIB_PATH=/usr/lib/python3/dist-packages/

# IMPORTANT: CC_OPT_FLAGS is the most important flag here, the very reason for taking the trouble to build
# Tensorflow from scratch.
#
# Set -march CPU architecture/tuning flag according to the /cpu/procinfo of the destination host where the docker container runs.
#
# See https://gcc.gnu.org/onlinedocs/gcc-5.4.0/gcc/x86-Options.html#x86-Options for list of architectures and
# which instruction sets they contain.
export CC_OPT_FLAGS='-march=haswell -mfpmath=both'

# This writes all the options to tools/bazel.rc which is picked up by bazel tool.
./configure

# Main build step.
bazel build --config=opt -k //tensorflow/tools/pip_package:build_pip_package

# Build a PIP wheel package.
mkdir -p ~/pip
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/pip

mv ~/pip /mnt/shared/tensorflow

echo "TENSORFLOW BUILD COMPLETED!"

