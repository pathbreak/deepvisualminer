from __future__ import print_function

import os
import os.path
import sys

sys.path.append('/root/darkflow')

from pipeline import Pipeline, MultiPipelineExecutor

# - Pipelines:
#   - deepdetect + basicdetect + facerecognize
#   - deepdetect only
#   - deepdetect + basicdetect
#   - basicdetect only
#   - basicdetect + facerecognize
#   - facetrain
#
# User inputs:
# - for detection/recognition:
#   - Input directory containing photos and videos
#   - Output directory for reports 
#   - Pipeline file
# 
# - for deep detection training
#   - TODO
#
# - for face recognition training
#   - input images directory
#   - size of training images

def detect(input_path, output_directory, pipeline_file):
    # if input_path is just a single file, we don't need all the multicore
    # setup.
    if os.path.isfile(input_path):
        pipeline = Pipeline(pipeline_file, os.path.dirname(input_path), output_directory)
        pipeline.execute(input_path)
        
    elif os.path.isdir(input_path):
        multiexecutor = MultiPipelineExecutor()
        multiexecutor.execute(pipeline_file, input_path, output_directory)
    
    
if __name__ == '__main__':
    detect(sys.argv[1], sys.argv[2], sys.argv[3])
