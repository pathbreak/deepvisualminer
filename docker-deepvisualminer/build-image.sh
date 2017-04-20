#!/bin/bash

sudo apt-get install -y python3 git

if [[ -d ./shared/darkflow && -f ./shared/darkflow/cython_utils/cy_yolo2_findboxes.cpython-35m-x86_64-linux-gnu.so ]]; then
    echo "Darkflow already seems to be built and copied to ./shared/darkflow. You have to delete ./shared/darkflow first if you want it rebuilt."
else

    mkdir -p ./shared

    echo "Launching Darkflow build container"
    sudo docker run  --rm --name darkflowbuild \
        -v $(pwd)/shared:/mnt/shared \
        -v $(pwd)/build-darkflow.sh:/root/build-darkflow.sh \
        ubuntu:16.04 \
        /root/build-darkflow.sh

    sudo chown -R $(logname):$(logname) ./shared

    # Download weight files shared by author via Google Drive.
    echo
    echo "Downloading pretrained weight files"
    python3 ./gdrive_downloader.py '0B1tW_VtY7oniZGlkLTh5YVl1WWs' ./shared/darkflow/cfg/yolo.weights
    python3 ./gdrive_downloader.py '0B1tW_VtY7oniTjM3YUxlRHpDVW8' ./shared/darkflow/cfg/tiny-yolo-voc.weights
fi

# Download OpenCV detector data files.
echo
if [[ -d ./opencv/data/haarcascades ]]; then
    echo "OpenCV detector data files already downloaded"
else
    echo "Downloading OpenCV detector data files"
    git clone https://github.com/opencv/opencv
fi
cd ./opencv
git pull
git fetch

cd ..
# Download my visualminer project
echo
if [[ -d ./deepvisualminer ]]; then
    echo "Deepvisualminer project already downloaded"
else
    echo "Downloading Deepvisualminer project"
    git clone https://github.com/pathbreak/deepvisualminer
fi
cd ./deepvisualminer
git pull
git fetch

cd ..


