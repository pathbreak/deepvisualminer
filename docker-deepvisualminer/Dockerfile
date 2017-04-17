FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-setuptools python3-pip ffmpeg && \
    pip3 install numpy opencv-python opencv-contrib-python tensorflow sklearn scipy imageio PyYAML simplejson && \
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
