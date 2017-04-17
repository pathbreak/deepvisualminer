from annotator import annotate
from basecomponent import BaseComponent

import imageio
import cv2
import os.path

class AnnotatedVideoWriter(BaseComponent):
    '''
    This is a outputter component that writes annotated frames to an output video.
    Unlike other components, design of this component has to be stateful - it opens output 
    video stream when it receives first frame and then keeps it open writing each frame to the video
    till completed notification is received.
    '''
    
    def __init__(self, cfg):
        BaseComponent.__init__(self, cfg)
        
        self.output_video = None
        self.output_filepath = None
        
        
    def execute(self, input_data, input_directory, output_directory):
        if not input_data['isvideo']:
            return {}
            
        # Open output video stream if this is first frame.
        if input_data['frame'] == 0:
            # The output directory structure should match input directory structure.
            relpath_of_input_file = os.path.relpath(input_data['file'], input_directory)
            relparent_of_input_file = os.path.dirname(relpath_of_input_file)
            inp_filename,inp_extension = os.path.splitext(os.path.basename(relpath_of_input_file))
            
            output_filedir = os.path.join(output_directory, relparent_of_input_file)
            if not os.path.exists(output_filedir):
                os.makedirs(output_filedir)
                
            self.output_filepath =  os.path.join(output_filedir,
                inp_filename + '-annotated.' + self.cfg['params']['format'])

            self.output_video = imageio.get_writer(self.output_filepath, 'ffmpeg')
            
            
            
        img = input_data['img'].copy()
        
        for comp in self.cfg['inputs']:
            comp_outputs = input_data.get(comp)
            comp_reports = comp_outputs['reports']
            if not comp_reports:
                print("Warning: pipeline file specifies {} as input for {} but {} is not outputting any location reports".format(
                    comp, self.name, comp
                ))
                continue
            
            annotate(img, comp_reports)
        
            
        final_img = cv2.resize(img, (self.cfg['params']['size']['width'], self.cfg['params']['size']['height']))
            
        self.output_video.append_data(final_img)
        
        return {'file': self.output_filepath}
                
                
    def completed(self, input_data, input_directory, output_directory):
        if self.output_video:
            self.output_video.close()
        
        self.output_video = None
        self.output_filepath = None
        
