from deepdetector import DeepDetector
from simpledetector import SimpleDetector
from facerecognizer import FaceRecognizer
from annotatedphotowriter import AnnotatedPhotoWriter
from annotatedframewriter import AnnotatedFrameWriter
from annotatedvideowriter import AnnotatedVideoWriter
from jsonreportwriter import JSONReportWriter


import imageio
import cv2

import multiprocessing
import os
import os.path
import yaml
import sys

class MultiPipelineExecutor(object):
    '''
    A multiprocess executor that sets up a pipeline for each CPU
    core and distributes input set of files across these pipelines.
    
    Each pipeline has its own detectors (including darkflow detectors), recognizer and outputter components,
    including  deep object detector neural networks. This may seem inefficient,
    but it appears darkflow stores some state per input file and is therefore not thread safe.
    '''
    def execute(self, pipeline_file, input_directory, output_directory):
        
        # Create a shared file queue across multiple processes.
        file_queue = multiprocessing.JoinableQueue()
        
        # Start pipelines.
        num_pipeline_processors = multiprocessing.cpu_count()
        num_pipeline_processors = 1
        print('Creating %d pipelines' % num_pipeline_processors)
        pipeline_processors = [ 
            PipelineProcessor(pipeline_file, input_directory, output_directory, file_queue) 
            for i in range(num_pipeline_processors) ]
            
        for w in pipeline_processors:
            w.start()
        
        # Enqueue files in input directory.
        for dirpath,dirs,files in os.walk(input_directory):
            for f in files:
                file_path = os.path.join(dirpath, f)
                print("put in queue:", file_path)
                file_queue.put(file_path)
        
        # Add an end command in each queue
        for i in range(num_pipeline_processors):
            file_queue.put(None)

        # Wait for all of the tasks to finish
        file_queue.join()
        
        print("Completed")
            
    
class PipelineProcessor(multiprocessing.Process):
    
    def __init__(self, pipeline_file, input_directory, output_directory, file_queue):
        multiprocessing.Process.__init__(self)

        self.file_queue = file_queue
        self.pipeline_file = pipeline_file
        self.input_directory = input_directory
        self.output_directory = output_directory
        
    def run(self):
        self.pipeline = Pipeline(self.pipeline_file, self.input_directory, self.output_directory)
        
        proc_name = self.name
        while True:
            next_file = self.file_queue.get()
            if next_file is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.file_queue.task_done()
                break
                
            print('%s: Executing %s' % (proc_name, next_file))
            self.pipeline.execute(next_file)
            print('%s: Executed %s' % (proc_name, next_file))
            self.file_queue.task_done()

        return        
    

class Pipeline(object):
    '''
    A Pipeline consists of a series of detectors, recognizers and outputters
    through which a photo or video is passed in a sequence.
    '''
    
    COMPONENTS = {
        'deepdetector' : DeepDetector,
        'simpledetector' : SimpleDetector,
        'photowriter' : AnnotatedPhotoWriter,
        'framewriter' : AnnotatedFrameWriter,
        'videowriter' : AnnotatedVideoWriter,
        'recognizer' : FaceRecognizer,
        'jsonreportwriter' : JSONReportWriter
    }
    
    def __init__(self, pipeline_file, input_directory, output_directory):
        
        with open(pipeline_file, 'r') as f:
            self.cfg = yaml.load(f)
            
        self.cfg = self.cfg['pipeline']
        self.output_directory = output_directory
        self.input_directory = input_directory
        
        self.create_components()
        
        
    def create_components(self):
        
        self.components = []
        
        for comp_cfg in self.cfg:
            comp_type = Pipeline.COMPONENTS.get(comp_cfg['type'])
            if comp_type:
                comp = comp_type(comp_cfg)
                self.components.append(comp)
            
            
    def execute(self, input_file):
        
        isphoto = False
        isvideo = False
        img = None
        
        # if input file is a photo, read it. Also create a separate
        # grayscale image because some of the detectors work on grayscale
        # but annotate on original color image. 
        # Send the color image, grayscale image, filepath, and isphoto flag
        # through the components of the pipeline.
        try:
            img = imageio.imread(input_file)
                
            print("Image read")
            
            isphoto = True            
            
        except:
            print("Not a photo. Error while attempting to load:", sys.exc_info())
            # If input file is a video, open it and setup an iterator over its
            # frames. Then for each frame, send image, grayscale image, video filename,
            # frame number and isvideo flag through the components of the pipeline.
            video = None
            try:
                video = imageio.get_reader(input_file, 'ffmpeg')
                print("Video opened")
                isvideo = True
            except:
                print("Not a video. Error while attempting to open:", sys.exc_info())
                if video:
                    video.close()
        
        if not isphoto and not isvideo:
            print("Ignoring file: ", input_file)
            return
            
        if isphoto:
            input_data = {
                'file' : input_file,
                'img' : img,
                'isphoto' : True,
                'isvideo' : False
            }
            
            self._execute_pipeline_on_image(input_data)
            
            self.completed(input_data)
            
        elif isvideo:
            
            for frame_num, img in enumerate(video):

                input_data = {
                    'file' : input_file,
                    'img' : img,
                    'isphoto' : False,
                    'isvideo' : True,
                    'frame' : frame_num
                }
                
                self._execute_pipeline_on_image(input_data)
            
            # Notify components such as video writers that need to know when
            # the input stream has completed so they can do their own cleanup.
            self.completed(input_data)
        
                
    def _execute_pipeline_on_image(self, input_data):
        
        if input_data['img'].ndim == 3:
            # It *appears* imageio imread returns RGB or RGBA, not BGR...confirmed using a blue
            # filled rectangle that imageio is indeed RGB which is opposite of OpenCV's default BGR.
            # Use RGB consistently everywhere.
            if input_data['img'].shape[-1] == 4:
                input_data['gray'] = cv2.cvtColor(input_data['img'], cv2.COLOR_RGBA2GRAY)
                print("Input image seems to be 4-channel RGBA. Creating 3-channel RGB version")
                input_data['img'] = cv2.cvtColor(input_data['img'], cv2.COLOR_RGBA2RGB)
            else:
                input_data['gray'] = cv2.cvtColor(input_data['img'], cv2.COLOR_RGB2GRAY)
            
        elif input_data['img'].ndim == 2:
            # If input is a grayscale image, it'll have just 2 dimensions, 
            # but Darkflow code expects 3 dimensions. So always keep 'img' a 3 dimension
            # image no matter what.
            print("Input image is grayscale. Creating RGB version")
            input_data['gray'] = input_data['img'].copy()
            input_data['img'] = cv2.cvtColor(input_data['img'], cv2.COLOR_GRAY2RGB)
            
        else:
            raise "Unknown image format " + input_data['img'].shape
        
        print("Input image:", input_data['img'].shape)
        
        for comp in self.components:
            print("Executing " + comp.name + " on " + input_data['file'])
            comp_outputs = comp.execute(input_data, self.input_directory, self.output_directory)
            
            # At each stage of the pipeline, collect the component's outputs
            # and add them to the input data so that they're available for 
            # downstream components.
            input_data[comp.name] = comp_outputs
            
        
        # Release the image arrays.
        input_data['img'] = None
        input_data['gray'] = None
            
            
    def completed(self, input_data):
        for comp in self.components:
            comp.completed(input_data, self.input_directory, self.output_directory)
