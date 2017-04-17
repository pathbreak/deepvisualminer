from basecomponent import BaseComponent

import os 
import os.path

import simplejson as json

class JSONReportWriter(BaseComponent):
    '''
    Component to write  detection and recognition regions and labels
    to a JSON file. The of each region is as specified in basecomponent.py.
    
    It writes 1 report per input file.
    
    The top level structure is a list of frames, regardless of photo or video.
    A photo is treated as a video with a single frame 0.
    In each frame, all the results of configured input sources are included, keyed by the
    component name.
    
    example:
    {
     'file': <filename>,
     'type': 'photo|video|,
     'frames': [
         {
            'frame' : 0,
            'coco-detector' : 
                [
                    {'labels':[{'label':'cat', 'confidence':0.8}, {'label':'lion', 'confidence':0.3}], 'rect':[x1,y1,x2,y2] },
                    {'labels':['dog','sheep'], 'rect':[x1,y1,x2,y2], 'confidence':0.8}
                ],
            ....
         }, 
         
         {
            'frame' : 1,
            'coco-detector' : 
                [
                    {'labels':[{'label':'cat', 'confidence':0.8}, {'label':'lion', 'confidence':0.3}], 'rect':[x1,y1,x2,y2] },
                    {'labels':['dog','sheep'], 'rect':[x1,y1,x2,y2], 'confidence':0.8}
                ],
            ...
            
         }
         ...
     ]
    }
    
    These results are cached in a dict till the completed notification is received, and then dumped
    to JSON.
    
    '''
    def __init__(self, cfg):
        BaseComponent.__init__(self, cfg)
        
        self.full_report = None
        
    def execute(self, input_data, input_directory, output_directory):
        if not self.full_report:
            self.full_report = {
                'file' : input_data['file'],
                'type' : 'photo' if input_data['isphoto'] else 'video',
                'frames' : []
            }
        
        frame_report = {
            'frame' : 0 if input_data['isphoto'] else input_data['frame'],
        }
        
        for comp in self.cfg['inputs']:
            comp_outputs = input_data.get(comp)
            comp_reports = comp_outputs['reports']
            
            frame_report[comp] = comp_reports
            
        self.full_report['frames'].append(frame_report)
        
        return {}

        
    def completed(self, input_data, input_directory, output_directory):
        
        # The output directory structure should match input directory structure.
        relpath_of_input_file = os.path.relpath(input_data['file'], input_directory)
        relparent_of_input_file = os.path.dirname(relpath_of_input_file)
        inp_filename,inp_extension = os.path.splitext(os.path.basename(relpath_of_input_file))
        
        output_filedir = os.path.join(output_directory, relparent_of_input_file)
        if not os.path.exists(output_filedir):
            os.makedirs(output_filedir)
            
        output_filepath =  os.path.join(output_filedir, inp_filename + '.json')
            
        print(output_filepath)
        
        
        with open(output_filepath, 'w') as f:
            json.dump(self.full_report, f, indent=4, separators=(',', ': '))

        
        return {'file':output_filepath}

        
