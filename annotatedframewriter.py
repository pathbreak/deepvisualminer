from annotator import annotate
from basecomponent import BaseComponent

import imageio
import cv2
import os.path

class AnnotatedFrameWriter(BaseComponent):
    def __init__(self, cfg):
        BaseComponent.__init__(self, cfg)
        
    def execute(self, input_data, input_directory, output_directory):
        if not input_data['isvideo']:
            return {}
        
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
        
        # The output directory structure should match input directory structure.
        relpath_of_input_file = os.path.relpath(input_data['file'], input_directory)
        relparent_of_input_file = os.path.dirname(relpath_of_input_file)
        inp_filename,inp_extension = os.path.splitext(os.path.basename(relpath_of_input_file))
        
        output_filedir = os.path.join(output_directory, relparent_of_input_file)
        if not os.path.exists(output_filedir):
            os.makedirs(output_filedir)
            
        output_filepath =  os.path.join(output_filedir,
            inp_filename + '-frame-' + str(input_data['frame']) + '-annotated.' + self.cfg['params']['format'])
            
        final_img = cv2.resize(img, (self.cfg['params']['size']['width'], self.cfg['params']['size']['height']))
            
        print(output_filepath)
        imageio.imwrite(output_filepath, final_img)
        
        return {'file':output_filepath}
                
        
