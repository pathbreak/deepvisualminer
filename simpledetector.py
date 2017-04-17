import cv2
from basecomponent import BaseComponent

class SimpleDetector(BaseComponent):
    '''
    A SimpleDetector uses Haar like or LBPH based cascade of weak classifiers
    for detection.
    '''
    
    def __init__(self, cfg):
        BaseComponent.__init__(self, cfg)
        
        params = cfg['params']
        
        self.detector = cv2.CascadeClassifier(params['model'])
        self.scaledown_factor = params.get('scaledown_factor', 1.1)
        self.min_neighbors = params.get('min_neighbors', 3)
        self.output_label = params['outputlabel']
        
        
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

        print(all_detections)
        
        results = {
            'reports' : all_detections
        }
        
        return results
        
        
    def detect_in_image(self, input_data):

        gray_img = input_data['gray']
        

        min_size = (min(50, gray_img.shape[0] // 10), min(50, gray_img.shape[1] // 10))
        
        # hits returned are (x,y,width, height) where x,y are topleft coordinates
        hits = self.detector.detectMultiScale(gray_img, 
            self.scaledown_factor, 
            self.min_neighbors, 
            0, min_size)
        
        results = [ 
            {
                'labels' : [{'label':self.output_label}],
                # these int casts are to avoid json serialization typerror because
                # the coords returned by opencv are actually np.int32s
                'rect':[int(x),int(y),int(x+w),int(y+h)]
            } for (x,y,w,h) in hits] 
            
        return results
            
        
    def detect_in_rois(self, input_data, comp_reports):
        gray_img = input_data['gray']

        min_size = (min(50, gray_img.shape[0] // 10), min(50, gray_img.shape[1] // 10))
        
        roi_detections = []
        
        for r in comp_reports:
            
            if ('all' in self.cfg['params']['triggerlabels']) or \
                any( [ l['label'] in self.cfg['params']['triggerlabels'] for l in r['labels'] ] ) :
            
                rect = r['rect']
                x_offset = rect[0]
                y_offset = rect[1]
                roi = gray_img[ rect[1]:rect[3], rect[0]:rect[2] ]
                
                hits = self.detector.detectMultiScale(gray_img, 
                    self.scaledown_factor, 
                    self.min_neighbors, 
                    0, min_size)

                # These detections in ROI are relative to ROI. So we must add ROI origin to
                # those coordinates to make them full image coordinates.
                results = [ 
                    {
                        'labels' : [{'label':self.output_label}],
                        # these int casts are to avoid json serialization typerror because
                        # the coords returned by opencv are actually np.int32s
                        'rect':[
                            int(x + x_offset),
                            int(y + y_offset),
                            int(x + x_offset + w),
                            int(y + y_offset + h)
                        ]
                    } for (x,y,w,h) in hits] 
                        
                
                roi_detections.extend(results)
            
        return roi_detections


