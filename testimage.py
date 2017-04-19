import tensorflow as tf

import cv2
import imageio

from deepdetector import DeepDetector
from simpledetector import SimpleDetector
from facerecognizer import FaceRecognizer

if __name__ == '__main__':
    hello = tf.constant('Hello, TensorFlow!')
    
    # Any warnings about CPU instructions are printed out when session 
    # is created
    sess = tf.Session()
