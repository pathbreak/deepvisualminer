from __future__ import print_function

import cv2
import cv2.face as face
import numpy as np

# skvideo doesn't seem to be able to handle some mp4 files
#from skvideo.io import vread, vreader
import imageio

import os
import os.path
import sys
import shutil

import csv
import json
from sklearn.model_selection import train_test_split

# Given a directory of original images with directory structure like this:
#  <top directory>
#     - label #1
#        - person1-image1
#        - person1-image2
#        - person1-image3
#        ...
#     - label #2
#        - person2-image1
#        - person2-image2
#        - person2-image3
#        ...
#     - label #3
#         ...
#
# this script can do the following tasks:
#
# - print statistics like mean and median dimensions of all images in entire dataset
#
# - scale all images to the same specified dimensions, either enlarge or shrink
#   and save them to a different location with the same directory structure
#
# - split an images directory into a pair of train and test directories
#
# - create a CSV file of image paths and labels from the directory structure
#
# - train a face recognizer using preferred algorithm and save the model for inference

def statistics(top_dir):
    widths = np.empty((0), dtype=np.uint16)
    heights = np.empty((0), dtype=np.uint16)
    
    for label in os.listdir(top_dir):
        label_dir = os.path.join(top_dir, label)
        for imgfilename in os.listdir(label_dir):
            imgfilepath = os.path.join(label_dir, imgfilename)
            img = cv2.imread(imgfilepath)
            
            widths = np.append(widths, img.shape[0])
            heights = np.append(heights, img.shape[1])
            
            
    mean_width = np.mean(widths)
    median_width = np.median(widths)
    width_hist = np.histogram(widths)
    print('Mean width=', mean_width)
    print('Median width=', median_width)
    print('Width histogram: ', width_hist)
    
    
    mean_height = np.mean(heights)
    median_height = np.median(heights)
    height_hist = np.histogram(heights)
    print('Mean height=', mean_height)
    print('Median height=', median_height)
    print('Height histogram: ', height_hist)



def scale(orig_top_dir, scaled_dest_dir, width, height, make_grayscale = True):
    
    if not os.path.exists(scaled_dest_dir):
        os.makedirs(scaled_dest_dir)
    
    for label in os.listdir(orig_top_dir):
        label_dir = os.path.join(orig_top_dir, label)
        dest_label_dir = os.path.join(scaled_dest_dir, label)
        
        if not os.path.exists(dest_label_dir):
            os.mkdir(dest_label_dir)
        
        for imgfilename in os.listdir(label_dir):
            orig_imgfilepath = os.path.join(label_dir, imgfilename)
            img = cv2.imread(orig_imgfilepath)
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            resized_gray = cv2.resize(gray_img, (width, height))
            
            dest_imgfilepath = os.path.join(dest_label_dir, imgfilename)
            
            cv2.imwrite(dest_imgfilepath, resized_gray)
            
 
        
def split_into_train_test_dirs(top_dir, train_dir, test_dir, train_percent):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for label in os.listdir(top_dir):
        label_dir = os.path.join(top_dir, label)
        
        images = os.listdir(label_dir)
        
        train_indexes = np.random.choice(np.arange(len(images)), int(train_percent * len(images) // 100), replace=False)
        
        for idx in xrange(len(images)):
            dest_dir = train_dir if idx in train_indexes else test_dir
            
            dest_label_dir = os.path.join(dest_dir, label)
            if not os.path.exists(dest_label_dir):
                os.mkdir(dest_label_dir)
            
            src_filename = os.path.join(label_dir, images[idx])
            
            shutil.copy(src_filename, dest_label_dir)
            
                

def export_csv(top_dir, dest_csv_file):
    
    with open(dest_csv_file, 'wb') as csvfile:
        labelwriter = csv.writer(csvfile, delimiter=',')
        
        for label_idx, label in enumerate(os.listdir(top_dir)):
            label_dir = os.path.join(top_dir, label)
            for imgfilename in os.listdir(label_dir):
                imgfilepath = os.path.abspath(os.path.join(label_dir, imgfilename))
                
                labelwriter.writerow([imgfilepath, label, label_idx])
    

def train(csv_file, train_percent, test_file_csv, models_dir, eigen=True, fischer=True, lbp=True):
    
    data = np.genfromtxt(csv_file,  delimiter=',', dtype=None, names=['file','label','labelnum'])
    label_counts = np.bincount(data['labelnum'])
    
    labels = {}
    for row in data:
        label = row[1]
        label_idx = row[2]
        if labels.get(label_idx) is None:
            labels[label_idx] = label

    # Every label should have atleast 2 data points. Delete those rows which don't 
    # satisfy that condition.
    data = data[ label_counts[data['labelnum']] >= 2 ]
    train_imagefiles, test_imagefiles = train_test_split(data, train_size=0.8, stratify=data['labelnum'])
    

    with open(test_file_csv, 'wb') as csvfile:
        testwriter = csv.writer(csvfile, delimiter=',')
        
        for test_imgfile in test_imagefiles:
            testwriter.writerow(list(test_imgfile))
    
    training_labels = train_imagefiles['labelnum']
    
    train_images = []
    for train_imgfile in train_imagefiles:
        train_images.append( cv2.imread(train_imgfile[0], cv2.IMREAD_GRAYSCALE) )

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    if eigen:
        eigen_recog = face.createEigenFaceRecognizer();
        eigen_recog.train(train_images, training_labels);
        eigen_recog.save(os.path.join(models_dir, 'eigen.yml'))
        print('Eigen done')
    
    if fischer:
        fischer_recog = face.createFisherFaceRecognizer();
        fischer_recog.train(train_images, training_labels);
        fischer_recog.save(os.path.join(models_dir, 'fischer.yml'))
        print('Fischer done')
    
    if lbp:
        lbp_recog = face.createLBPHFaceRecognizer();
        lbp_recog.train(train_images, training_labels);
        lbp_recog.save(os.path.join(models_dir, 'lbp.yml'))
        print('LBP done')
    
    # Record the training image dimensions because at prediction time we need to resize images 
    # to those dimensions.
    model = {'width' : train_images[0].shape[1], 'height' : train_images[0].shape[0], 'labels' : labels}
    with open(os.path.join(models_dir, 'model.json'), 'w') as model_file:
        json.dump(model, model_file, indent=4, separators=(',', ': '))
    

def recognize(img_file, expected_label, models_dir, eigen=True, fischer=True, lbp=True):

    eigen_label = fischer_label = lbp_label = -1

    with open(os.path.join(models_dir, 'model.json'), 'r') as model_file:
        model = json.load(model_file)
        train_img_size = (model['height'], model['width'])
       
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if img.shape != train_img_size:
        img = cv2.resize( img, train_img_size[::-1] )
    
    if eigen:
        eigen_recog = face.createEigenFaceRecognizer();
        eigen_recog.load(os.path.join(models_dir, 'eigen.yml'))
        eigen_label = eigen_recog.predict(img)
        print('Eigen done')
    
    if fischer:
        fischer_recog = face.createFisherFaceRecognizer();
        fischer_recog.load(os.path.join(models_dir, 'fischer.yml'))
        fischer_label = fischer_recog.predict(img)
        print('Fischer done')
    
    if lbp:
        lbp_recog = face.createLBPHFaceRecognizer();
        lbp_recog.load(os.path.join(models_dir, 'lbp.yml'))
        lbp_label = lbp_recog.predict(img)
        print('LBP done')
    
    
    print(eigen_label, fischer_label, lbp_label)
    return  eigen_label, fischer_label, lbp_label




def test(test_csv, models_dir, eigen=True, fischer=True, lbp=True):

    eigen_label = fischer_label = lbp_label = -1

    if eigen:
        eigen_recog = face.createEigenFaceRecognizer();
        eigen_recog.load(os.path.join(models_dir, 'eigen.yml'))

    if fischer:
        fischer_recog = face.createFisherFaceRecognizer();
        fischer_recog.load(os.path.join(models_dir, 'fischer.yml'))

    if lbp:
        lbp_recog = face.createLBPHFaceRecognizer();
        lbp_recog.load(os.path.join(models_dir, 'lbp.yml'))
    
    with open(os.path.join(models_dir, 'model.json'), 'r') as model_file:
        train_img_size = json.load(model_file)
        train_img_size = (train_img_size['height'], train_img_size['width'])
       
    test_imgfiles = np.genfromtxt(test_csv,  delimiter=',', dtype=None, names=['file','label','labelnum'])
    
    for test_imgfile in test_imgfiles:
        
        img = cv2.imread(test_imgfile[0], cv2.IMREAD_GRAYSCALE)

        if img.shape != train_img_size:
            img = cv2.resize( img, train_img_size[::-1] )

        expected_label = test_imgfile[2]
        
        if eigen:
            eigen_label = eigen_recog.predict(img)
    
        if fischer:
            fischer_label = fischer_recog.predict(img)
  
        if lbp:
            lbp_label = lbp_recog.predict(img)

        print(test_imgfile[0], expected_label, eigen_label, fischer_label, lbp_label)



def detect(img_file, detector_xml_path, dest_img_file):
    img = cv2.imread(img_file)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier(detector_xml_path)
    
    min_size = (min(50, gray_img.shape[0] // 10), min(50, gray_img.shape[1] // 10))
    hits = detector.detectMultiScale(gray_img, 1.1, 4, 0, min_size)
    #cv2.groupRectangles(hits, 2)
    print(hits)
    
    hits_img = np.copy(img)
    for (x,y,w,h) in hits:
        cv2.rectangle(hits_img, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.imwrite(dest_img_file, hits_img)



def detectvideo(vid_file, detector_xml_path, dest_img_dir):
    
    if not os.path.exists(dest_img_dir):
        os.makedirs(dest_img_dir)

    detector = cv2.CascadeClassifier(detector_xml_path)
    
    vid = imageio.get_reader(vid_file, 'ffmpeg')
    # If size and source_size are not equal, then device was probably
    # rotated (like a mobile) and we should compensate for the rotation.
    # Images will have 'source_size' dimensions but we need 'size'.
    metadata = vid.get_meta_data()
    rotate = False
    if metadata['source_size'] != metadata['size']:
        print('Rotating')
        rotate = True
    
    for i, img in enumerate(vid):
        if rotate:
            #img = np.transpose(img, axes=(1, 0, 2)).copy()
            img = np.rot90(img).copy()
            
        print('Frame ',i, img.shape)
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        min_size = (min(20, gray_img.shape[0] // 10), min(20, gray_img.shape[1] // 10))
        hits = detector.detectMultiScale(gray_img, 1.1, 3, 0, min_size)
        #cv2.groupRectangles(hits, 2)
        print(len(hits), ' hits')

        hits_img = np.copy(img)
        
        if len(hits) > 0:
            for (x,y,w,h) in hits:
                cv2.rectangle(hits_img, (x,y), (x+w, y+h), (0,0,255), 2)

        cv2.imwrite(os.path.join(dest_img_dir, 'frame-%d.png'%(i)), hits_img)





def recognizemany(img_file, detector_xml_path, models_dir, dest_img_file, eigen=True, fischer=True, lbp=True):

    img = cv2.imread(img_file)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #gray_img = cv2.resize(gray_img, (640, 480))
    
    detector = cv2.CascadeClassifier(detector_xml_path)
    
    min_size = (min(50, gray_img.shape[0] // 10), min(50, gray_img.shape[1] // 10))
    #min_size = (0,0)
    hits = detector.detectMultiScale(gray_img, 1.1, 3, 0, min_size)

    eigen_label = fischer_label = lbp_label = -1

    with open(os.path.join(models_dir, 'model.json'), 'r') as model_file:
        model = json.load(model_file)
        train_img_size = (model['height'], model['width'])
        labels = model['labels']
    
    print('# hits:', len(hits))
    
    hits_img = np.copy(img)
    
    i = 1
    for (x,y,w,h) in hits:
        print('ROI ', i)
        roi = gray_img[y:y+h, x:x+w]
        i += 1

        if roi.shape != train_img_size:
            roi = cv2.resize( roi, train_img_size[::-1] )
    
        if eigen:
            eigen_recog = face.createEigenFaceRecognizer();
            eigen_recog.load(os.path.join(models_dir, 'eigen.yml'))
            eigen_label = eigen_recog.predict(roi)
            print('Eigen done')
        
        if fischer:
            fischer_recog = face.createFisherFaceRecognizer();
            fischer_recog.load(os.path.join(models_dir, 'fischer.yml'))
            fischer_label = fischer_recog.predict(roi)
            print('Fischer done')
        
        if lbp:
            lbp_recog = face.createLBPHFaceRecognizer();
            lbp_recog.load(os.path.join(models_dir, 'lbp.yml'))
            lbp_label = lbp_recog.predict(roi)
            print('LBP done')

        cv2.rectangle(hits_img, (x,y), (x+w, y+h), (255,255,255), 2)
        cv2.putText(hits_img,  labels[str(fischer_label)], (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 2)
            
    
        print(labels[str(eigen_label)], labels[str(fischer_label)], labels[str(lbp_label)])
        #return  eigen_label, fischer_label, lbp_label

    
    cv2.imwrite(dest_img_file, hits_img)

    
    

#########################################3

if __name__ == '__main__':
    if sys.argv[1] == 'stats':
        statistics(sys.argv[2])
 
    elif sys.argv[1] == 'resize':
        scale( sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), bool(sys.argv[6]) )

    elif sys.argv[1] == 'split':
        split_into_train_test_dirs( sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) )
        
    elif sys.argv[1] == 'csv':
        export_csv( sys.argv[2], sys.argv[3])
        
    elif sys.argv[1] == 'train':
        train( sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], bool(sys.argv[6]), bool(sys.argv[7]), bool(sys.argv[8]) )
        
    elif sys.argv[1] == 'test':
        test( sys.argv[2], sys.argv[3], bool(sys.argv[4]), bool(sys.argv[5]), bool(sys.argv[6]) )
        
    elif sys.argv[1] == 'recognize':
        recognize( sys.argv[2], int(sys.argv[3]), sys.argv[4], bool(sys.argv[5]), bool(sys.argv[6]), bool(sys.argv[7]) )
        
    elif sys.argv[1] == 'detect':
        detect( sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == 'recognizemany':
        recognizemany( sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], bool(sys.argv[6]), bool(sys.argv[7]), bool(sys.argv[8]) )
        
    elif sys.argv[1] == 'detectvideo':
        detectvideo( sys.argv[2], sys.argv[3], sys.argv[4])
        
