# Deep Visual Miner for photos and videos

Deep visual mining for your photos and videos using YOLOv2 deep convolutional neural network based object detector 
and traditional face recognition algorithms.


## Hardware selection

This software is designed 


## Install Docker

The software along with all its dependencies are packaged in a Docker image so that it can
be deployed on any distribution without hassle.

For running it, first you have to install Docker.

Click the link to your OS or distribution in the [Docker installation page](https://docs.docker.com/engine/installation/) 
for instructions to install Docker-CE.

For Ubuntu 16.04, these steps taken from [https://docs.docker.com/engine/installation/linux/ubuntu/]:

 ```
 sudo apt-get install  apt-transport-https  ca-certificates  curl  software-properties-common

 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

 sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
   
 sudo apt-get update

 sudo apt-get install docker-ce
 ```



## Download the Docker images

For Haswell and later architecture machines, pull the image that contains optimized TensorFlow:

 ```
   docker pull pathbreak/deepvisualminer-haswell
   
 ```
 
For other x86-64 architecture machines, use the most compatible version that should work everywhere:

 ```
   docker pull pathbreak/deepvisualminer
   
 ```



## Build your own Docker images

If you don't want to pull my Docker images published at https://hub.docker.com/r/pathbreak/deepvisualminer/
and https://hub.docker.com/r/pathbreak/deepvisualminer-haswell/,
you can build one or both on your own...

1. Install Docker on the machine where you'll be building the images. See [Install Docker](#install-docker) 
   for steps.

2. Clone this project repo from GitHub:

 ```
   git clone https://github.com/pathbreak/deepvisualminer
   cd deepvisualminer
   git checkout master 
   cd docker-deepvisualminer
   chmod +x *.sh
 ```
   The files required for building docker image are under `docker-deepvisualminer/` subdirectory.
 
3. Next, decide whether you want to build an optimized version of TensorFlow or use the stock
   version available in PyPI. If you want to use the stock version, skip the rest of this section 
   and jump to step 3 directly.
   
   The optimized version is for optimized only for running on CPUs and is potentially faster at linear algebra
   operations because it uses modern CPU instruction sets like SSE4.x, SSSE3, AVX, AVX2, FMA, etc.
   If does **not** include GPU support (but you can enable it easily by adding '--config=cuda' and TF_NEED_CUDA=1
   in `build-tensorflow.sh` - note that neither these scripts nor images have not been tested for CUDA configuration).
   
   If you decide to build optimized version, see 
   https://gcc.gnu.org/onlinedocs/gcc-5.4.0/gcc/x86-Options.html#x86-Options for list of architectures and
   which instruction sets they contain, and edit the "CC_OPT_FLAGS='-march=<whatever>..." line appropriately in 
   `docker-deepvisualminer/build-tensorflow.sh`.
   
   Then run `launch-tensorflow.sh` to launch a container in which optimized TensorFlow gets built from
   source code. Beware: this takes a long time - 1 to 1.5 hours - and occupies a lot of RAM and CPU. That said,
   I've built it on a Core-i3 8GB machine with no problems or pauses...
   
 ```
    ./launch-tensorflow.sh
 ```
 
   The optimized TensorFlow package is placed under `./shared/tensorflow/`.
   
4. Next, run `docker-deepvisualminer/build-image.sh` that builds or downloads other prerequisites.   
   It builds Darkflow in a temporary container and downloads OpenCV detector data files:

 ```
  ./build-image.sh
 ```
 
  This command launches a temporary container to build the Darkflow object detection project.
  It's done in a separate container to avoid bloating up the primary deepvisualminer docker images
  with build tools and artifacts.
  
  This script also downloads pretrained neural network weight files shared by Darkflow's author via Google Drive.
  
  Once the build is complete, the complete Darkflow release directory is copied under host system's `./shared/darkflow` directory.
  
  It downloads OpenCV cascade classifier pretrained models to `./opencv/data`.
  
  It downloads all the python scripts of this project in another subdirectory to copy them into the image.
  
5. Finally build the images.

   If you had selected to build optimized version of TensorFlow, build the optimized image 
   using the Dockerfile `docker-deepvisualminer/Dockerfile-custom-tensorflow`. For example:

 ```
 sudo docker build -t deepvisualminer-optimized -f Dockerfile-custom-tensorflow .
 ```

   If you had gone with the stock version of TensorFlow, build the stock image using
   the Dockerfile `docker-deepvisualminer/Dockerfile-stock-tensorflow`:

 ```
 sudo docker build -t deepvisualminer -f Dockerfile-stock-tensorflow .
 ```

6. Verify that the image is created:

 ```
 sudo docker images
 ```
 
 
 
## Test Docker images

In order to test whether the Docker image you downloaded works on a target machine's CPU,
or to test whether one you built yourself is correct, start a test container and follow these steps:

 ```
 sudo docker run -ti --rm --name testdeepvisualminer deepvisualminer bash
 ```
 
 where `deepvisualminer` is the image name you downloaded (it'll be `deepvisualminer-haswell` if
 you downloaded the optimized version, or whatever name you gave if you built the image yourself.
 
 In the container shell, run this:
 ```
 cd ~/deepvisualminer
 python3 testimage.py
 ```
 If everything is ok, there shouldn't be any errors or warnings. If you see `Illegal instruction` errors,
 you've downloaded the wrong image or built one for a higher architecture by mistake.
 
 
 
## Train the face recognizer

The face recognizer has to be "trained" before it can recognize any faces.

Training basically means the recognizer runs an algorithm on a set of facial images to
work out a mathematical model of the faces. It uses this model later during recognition 
to find out which face it saw during training resembles the input face best.

Steps are:

+ Go through your photo collections and select a subset of photos containing all 
  the individuals whose faces you want the system to recognize. 
  
  Select minimum 15 photos for each individual. 
  The more photos you use for training, the more accurate recognition will be.
  
  The system can't use videos for training. If a photo you want is inside a video,
  use a screenshot tool (Shutter on Ubuntu is excellent) or frame extractor (ffmpeg can do it)
  to save that frame as an image.

+ Do not mix species! If you want the system to recognize people and cats, 
  create one subset containing people and a separate one containing cats, 
  and train them separately as two different models.

+ The poses should match the kind of recognition you want. If you want people to be recognized 
  from a frontal face photo, select frontal face photos for training too; if you want 
  side profile recognition, select side profile photos.
  
+ Use an image editor like Gimp or Photoshop to crop only the facial areas, 
  and export the cropped areas as PNG files. Try to exclude any area not part of the 
  face - such as neck, clothes, hair, background.
  
  There's no need to manually resize all cropped images to the same size - you risk losing 
  useful information that way. The script is capable of statistically analyzing all images in the 
  dataset and resizing them all to ideal dimensions.
  
  Save the cropped images in a directory structure like this:
  
 ```
  \peoplefrontfaces
     \person-1
        image1.png
        image2.png
        ...
        
     \person-2
        image1.png
        image2.png
        ...
        
  \catfrontfaces
     \cat-1
        image1.png
        image2.png
        ...
        
     \cat-2
        image1.png
        image2.png
        ...
    
 ```
  Here, 'person-1', 'cat-1', etc should be their actual names because they are
  output in reports and annotated images.
  
+ Run a statistical analysis on the set of facial image to find out an ideal resizing
  dimension. All training and testing images should be of the same size for the math
  to work correctly. Even during recognition, the detected facial region should be
  resized to this same image size for recognition to work at all.
  
  ```
  sudo docker run --rm -v [TOP-DIR-OF-IMAGES]:/root/images   deepvisualminer \
    python3 /root/deepvisualminer/facerec_train.py stats /root/images
  ```
