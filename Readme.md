# Deep Visual Miner for photos and videos

Deep visual mining for your photos and videos using YOLOv2 deep convolutional neural network based object detector 
and traditional face recognition algorithms.

## Download the Docker image

 ```
   docker pull pathbreak/deepvisualminer
   
 ```

## Build your own Visual Miner Docker image

If you don't want to pull my Docker image published at https://hub.docker.com/r/pathbreak/deepvisualminer/,
you can build it on your own machine locally.

1. Clone deepvisualminer project repo from GitHub:

 ```
 git clone https://github.com/pathbreak/deepvisualminer
 ```
 
 The files required for building docker image are in `docker-deepvisualminer` subdirectory:
 
2. First, you need to build or download prerequisites. 
   The following script builds Darkflow in a temporary container and downloads OpenCV detector data files:

 ```
 cd deepvisualminer/docker-deepvisualminer
 chmod +x *.sh
 sudo ./build-image.sh
 ```
 
  This command launches a temporary container just to build the Darkflow object detection project.
  
  It's done in a separate container to avoid bloating up the primary deepvisualminer docker image
  with build tools and artifacts.
  
  This script also downloads pretrained neural network weight files shared by author via Google Drive.
  
  Once the build is complete, the complete Darkflow release directory is copied under host system's `./host/` directory.
  
3. Finally build the primary deepvisualminer docker image.

 ```
 sudo docker build -t deepvisualminer .
 ```

4. Verify that the image is created.

 ```
 sudo docker images
 ```
