import os
import cv2
import numpy as np

def algorithm(input_file =  None, binary_threshold = 180, outlier_detection = 'z-score', mask_area_thr = 640000):

    image = cv2.imread(input_file)

    # The initial processing of the image

    image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting

    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(image_bw)

    # 2 erosions and 1 dilation

    final_img=cv2.erode(final_img,None,iterations=2)
    final_img=cv2.dilate(final_img,None,iterations=1)

    

    # Ordinary thresholding the same image

    _, ordinary_img = cv2.threshold(final_img, binary_threshold, 255, cv2.THRESH_BINARY)
 
    ordinary_img=cv2.erode(ordinary_img,None,iterations=1)
    ordinary_img=cv2.dilate(ordinary_img,None,iterations=1)

    mask_areas=[]
    canvases=[]
    contours_draw,hierarchy=cv2.findContours(ordinary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    #save mask areas and masks

    for i in range(len(contours_draw)):
        canvas=np.zeros(image.shape[0:2],np.uint8)
        cv2.drawContours(canvas,[contours_draw[i]],-1,(255,255,255),-1)
        mask_areas.append(canvas.sum()/255)
        canvases.append(canvas)

    indexes=[]

    #outlier detection through z-score or tukeys method

    if(outlier_detection=='z-score'):
        thr=np.mean(mask_areas)+3*np.std(mask_areas)
        for i in range(len(mask_areas)):
            if(mask_areas[i]>thr or mask_areas[i]>mask_area_thr):
                indexes.append(i)
    else:
        thr=np.percentile(mask_areas,75) + 1.5*(np.percentile(mask_areas,75)-np.percentile(mask_areas,25))
        for i in range(len(mask_areas)):
            if(mask_areas[i]>thr or mask_areas[i]>mask_area_thr):
                indexes.append(i)
    
    outlier_removed=ordinary_img

    for i in indexes:
        outlier_removed=cv2.bitwise_xor(canvases[i],outlier_removed)
    
    outlier_removed=cv2.erode(outlier_removed,None,iterations=1)

    #post_processing
    edge_map = cv2.Canny(final_img, 120, 250)

    edge_contours = sorted(cv2.findContours(edge_map,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2], key = cv2.contourArea)[:-2]
    new_mask_M = np.zeros( edge_map.shape, dtype="uint8" )
    drawn_edge_contours = cv2.drawContours(new_mask_M, edge_contours,-1,(255,255,255),20)

    post_processed = cv2.bitwise_and(outlier_removed, drawn_edge_contours)

    return post_processed