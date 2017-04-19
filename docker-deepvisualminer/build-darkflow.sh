#!/bin/bash

apt-get update

apt-get install -y python3 python3-pip git

pip3 install Cython numpy

cd ~

git clone https://github.com/pathbreak/darkflow
cd darkflow
git fetch
git checkout pathbreak

cd cython_utils

python3 setup.py build_ext --inplace

rm -rf ~/darkflow/.git

mv ~/darkflow /mnt/shared/

echo "DARKFLOW BUILD COMPLETED!"

