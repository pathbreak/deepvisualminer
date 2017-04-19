# Dockerfile for deepvisualminer image with all required dependencies.
#
# The TensorFlow installed is a custom build with support for 'haswell' architecture instruction
# sets -  MOVBE, MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, AVX, AVX2, AES, PCLMUL, FSGSBASE, RDRND, FMA, BMI, BMI2 and F16C.
# See https://gcc.gnu.org/onlinedocs/gcc-5.4.0/gcc/x86-Options.html#x86-Options for all available architectures.
#
# Check /proc/cpuinfo that these instructions are available in the docker host before running
# a container off this image. Otherwise, it'll result in illegal instruction error.

FROM ubuntu:16.04

# launch-tensorflow.sh should have been executed prior to this.
COPY ./shared/tensorflow /root/tensorflow

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-setuptools python3-pip ffmpeg && \
    pip3 install --upgrade pip && \
    pip3 install numpy opencv-python opencv-contrib-python sklearn scipy imageio PyYAML simplejson && \ 
    pip3 install /root/tensorflow/* && \
    apt-get remove -y python3-pip python3-setuptools && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt/* && \
    rm -rf /root/.cache/pip/*

# build-image.sh should have been launched prior to this.
COPY ./shared/darkflow /root/darkflow

COPY ./opencv/data /root/data

# TODO Install the video mining scripts
COPY ./deepvisualminer /root/deepvisualminer

#ADD ./my_service.py /my_service.py
#ENTRYPOINT ["python", "/my_service.py"]
