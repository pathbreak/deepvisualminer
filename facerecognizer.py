import cv2
import cv2.face as face
import numpy as np

from basecomponent import BaseComponent

import os.path
import simplejson as json

class FaceRecognizer(BaseComponent):
    '''
    A FaceRecognizer uses Eigen, Fisher, LBPH or combination of them
    for face recognition.
    '''
    
    def __init__(self, cfg):
        BaseComponent.__init__(self, cfg)
         
        params = cfg['params']
        
        models_dir = params['model']
        if not os.path.exists(models_dir):
            raise "Error: Invalid face recognizer model directory path " + models_dir
        
        strategies = params['strategies']
        if not strategies:
            raise "Error: Invalid pipeline file. Recognizer should specify atleast 1 strategy: eigen|fischer|lbp"
        
        self.output_label = params['outputlabel']

        if 'eigen' in strategies:
            self.eigen = face.createEigenFaceRecognizer();
            self.eigen.load(os.path.join(models_dir, 'eigen.yml'))
        else:
            if 'eigen' in self.output_label:
                raise "Error: Invalid pipeline file. Recognizer has eigen in output label but not in strategies"
                
            self.eigen = None
        
        if 'fischer' in strategies:
            self.fischer = face.createFisherFaceRecognizer();
            self.fischer.load(os.path.join(models_dir, 'fischer.yml'))
        else:
            if 'fischer' in self.output_label:
                raise "Error: Invalid pipeline file. Recognizer has fischer in output label but not in strategies"
                
            self.fischer = None
        
        if 'lbp' in strategies:
            self.lbp = face.createLBPHFaceRecognizer();
            self.lbp.load(os.path.join(models_dir, 'lbp.yml'))
        else:
            if 'lbp' in self.output_label:
                raise "Error: Invalid pipeline file. Recognizer has lbp in output label but not in strategies"
                
            self.lbp = None
        
        with open(os.path.join(models_dir, 'model.json'), 'r') as model_file:
            self.model = json.load(model_file)
            self.train_img_size = (self.model['height'], self.model['width'])
            self.labels = self.model['labels']
        
        self.equalize_hist = params.get('equalizehist', False)
        
        
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
                    detections = self.detect_in_rois(input_data, comp_reports)
                    all_detections.extend(detections)

        print(all_detections)

        results = {
            'reports' : all_detections
        }
        
        return results
        
    
    def detect_in_image(self, input_data):
        
        gray_img = input_data['gray']
        
        results = self._detect_in_area(gray_img.copy())
        
        return results


    def detect_in_rois(self, input_data, comp_reports):
        print("Facerecognizer: detect in ROIs")
        gray_img = input_data['gray']
        print("Facerecognizer: gray image", gray_img.shape)

        roi_detections = []
        
        for r in comp_reports:
            
            if ('all' in self.cfg['params']['triggerlabels']) or \
                any( [ l['label'] in self.cfg['params']['triggerlabels'] for l in r['labels'] ] ) :
                
                rect = r['rect']
                print("Facerecognizer: ROI rect ", rect)
                x_offset = rect[0]
                y_offset = rect[1]
                roi = gray_img[ rect[1]:rect[3], rect[0]:rect[2] ]
                print("Facerecognizer: ROI ", roi.shape)
                
                results = self._detect_in_area(roi.copy())
                
                for r in results:
                    rect = r['rect']
                    rect[0] += x_offset
                    rect[1] += y_offset
                    rect[2] += x_offset
                    rect[3] += y_offset
                
                roi_detections.extend(results)
                
        return roi_detections
        
                
                
                
        
    def _detect_in_area(self, gray_img):

        if self.equalize_hist:
            gray_img = cv2.equalizeHist(gray_img)
        
        if gray_img.shape != self.train_img_size:
            print("recognizer input, trainings sizes:", gray_img.shape, self.train_img_size[::-1])
            roi = cv2.resize( gray_img, (self.train_img_size[1], self.train_img_size[0]) )
        else:
            roi = gray_img
            
            
        eigen_label, eigen_conf = self.eigen.predict(roi) if self.eigen else (-1,-1)
        fischer_label, fischer_conf = self.fischer.predict(roi) if self.fischer else (-1,-1)
        lbp_label,lbp_conf = self.lbp.predict(roi) if self.lbp else (-1,-1)
        
        labels = []
        if 'all' in self.output_label:
            if self.eigen:
                labels.append({'label':self.labels[str(eigen_label)], 'method':'eigen'})
            if self.fischer:
                labels.append({'label':self.labels[str(fischer_label)], 'method':'fischer'})
            if self.lbp:
                labels.append({'label':self.labels[str(lbp_label)], 'method':'lbp'})
            
        elif 'mostvotes' in self.output_label:
            methods = ['eigen', 'fischer', 'lbp']
            label_values = [eigen_label, fischer_label, lbp_label]
            votes = [label_values.count(l) if l >= 0 else 0 for l in label_values]
            most_votes = max(votes)
            most_voted_label_value = label_values[np.argmax(most_votes)]
            most_votes_methods = []
            for i,m in enumerate(methods):
                if votes[i] == most_votes:
                    most_votes_methods.append(m)
            labels.append({'label':self.labels[str(most_voted_label_value)], 'method':','.join(most_votes_methods)})
            
        elif 'eigen' in self.output_label:
            labels.append({'label':self.labels[str(eigen_label)], 'method':'eigen'})
            
        elif 'fischer' in self.output_label:
            labels.append({'label':self.labels[str(fischer_label)], 'method':'fischer'})
            
        elif 'lbp' in self.output_label:
            labels.append({'label':self.labels[str(lbp_label)], 'method':'lbp'})
                

        results = [ 
            {
                'labels' : labels,
                'rect':[0,0,gray_img.shape[1],gray_img.shape[0]]
            } ] 
            
        return results
        
        
        

