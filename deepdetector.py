import sys
sys.path.append('/root/darkflow')

from net.build import TFNet
from basecomponent import BaseComponent
from annotator import annotate

class DeepDetector(BaseComponent):
    '''
    A DeepDetector uses a YOLOv2 convolutional neural network model for
    object detection.
    '''
    
    def __init__(self, cfg):
        BaseComponent.__init__(self, cfg)
        
        params = self.cfg['params']
        
        tfnet_cfg = {
            "model": params['model'],
            "load": params['weights'], 
            "config" : '/root/darkflow/cfg',
            "verbalise" : True,
            "threshold": 0.1
        }
        
        self.nn = TFNet(tfnet_cfg)
        
        
    def execute(self, input_data, input_directory, output_directory):
        
        # Check what configured inputs are - whether complete image or ROIs output by some
        # other components.
        all_detections = []
        for source in self.cfg['inputs']:
            if source == 'files':
                detections = self.detect_in_image(input_data)
                all_detections.extend(detections)
                
            else:
                triggerlabels = self.cfg['params'].get('triggerlabels')
                if not triggerlabels:
                    print("Warning: pipeline file specifies {} in inputs but there are no triggerlabels in params".format(source))
                    continue
                    
                comp_outputs = input_data.get(source)
                if comp_outputs:
                    comp_reports = comp_outputs['reports']
                    detections = self.detect_in_rois(self, input_data, comp_reports)
                    all_detections.extend(detections)
        
        # Each detection is of the form 
        # {"label":"person", "confidence": 0.56, "topleft": {"x": 184, "y": 101}, "bottomright": {"x": 274, "y": 382}}
        # These should be transformed to our preferred JSON output documented in basecomponent.py
        
        reports = []
        for d in all_detections:
            r = {
                'labels' : [
                    {
                        'label' : d['label'],
                        # The float() here is because that confidence value is actually a np.float32
                        # and that creates serialization typeerror problems while writing report to
                        # json.
                        'confidence' : float(d['confidence'])
                    }
                ],
                'rect' : [
                    d['topleft']['x'],
                    d['topleft']['y'],
                    d['bottomright']['x'],
                    d['bottomright']['y'],
                ]
            }
            
            reports.append(r)
            
        results = {
            'reports' : reports
        }
        
           
        print(results)
        return results
        
        


    def detect_in_image(self, input_data):
        print("Deep detector starting " + input_data['file'])
        detections = self.nn.return_predict(input_data['img'])
        print("Deep detector completed"  + input_data['file'])
        return detections




    def detect_in_rois(self, input_data, comp_reports):
        img = input_data['img']
        roi_detections = []
        
        for r in comp_reports:
            
            if ('all' in self.cfg['params']['triggerlabels']) or \
                any( [ l['label'] in self.cfg['params']['triggerlabels'] for l in r['labels'] ] ) :
            
                rect = r['rect']
                x_offset = rect[0]
                y_offset = rect[1]
                roi = img[ rect[1]:rect[3], rect[0]:rect[2], :]
                
                detections = self.nn.return_predict(roi)
                # These detections in ROI are relative to ROI. So we must add ROI origin to
                # those coordinates to make them full image coordinates.
                for d in detections:
                    d['topleft']['x'] += x_offset
                    d['bottomright']['x'] += x_offset
                    
                    d['topleft']['y'] += y_offset
                    d['bottomright']['y'] += y_offset
                
                roi_detections.extend(detections)
            
        return roi_detections
