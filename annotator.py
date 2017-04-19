import cv2

def annotate(img, comp_reports):
    '''
    Draws boxes and label text around a detection region.
    Caution: Since img is overwritten, ensure caller passes a copy instead of original image.
    '''
    for r in comp_reports:
        rect = r['rect']
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255,255,255), 2)
        
        # Position the text annotation above rectangle by default, unless rectangle is at border.
        text_y = rect[1]-5 if rect[1] >= 5 else rect[1]+5
        cv2.putText(img,  r['labels'][0]['label'], (rect[0], text_y), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 2)
