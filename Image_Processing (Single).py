import os
import cv2
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--input_file', type=str, help='input file')
parser.add_argument('--binary_threshold', type=int, default=180, help='input file')
parser.add_argument('--outlier_detection', type=str, default='z-score', help='z-score/tukeys method')
parser.add_argument('--mask_area_thr', type=int, default=640000, help='additional threshold for outlier detection')
parser.add_argument('--output_folder', type=str, help='input file')

args = parser.parse_args()

# Reading the image from the present directory

image = cv2.imread(args.input_file)

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

_, ordinary_img = cv2.threshold(final_img, args.binary_threshold, 255, cv2.THRESH_BINARY)
 
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

if(args.outlier_detection=='z-score'):
    thr=np.mean(mask_areas)+3*np.std(mask_areas)
    for i in range(len(mask_areas)):
        if(mask_areas[i]>thr or mask_areas[i]>args.mask_area_thr):
            indexes.append(i)
else:
    thr=np.percentile(mask_areas,75) + 1.5*(np.percentile(mask_areas,75)-np.percentile(mask_areas,25))
    for i in range(len(mask_areas)):
        if(mask_areas[i]>thr or mask_areas[i]>args.mask_area_thr):
            indexes.append(i)
    
outlier_removed=ordinary_img

for i in indexes:
    outlier_removed=cv2.bitwise_xor(canvases[i],outlier_removed)
    
outlier_removed=cv2.erode(outlier_removed,None,iterations=1)

#im=cv2.dilate(im,None,iterations=1)

cv2.imshow("CLAHE",final_img)
cv2.imshow("ordinary threshold", ordinary_img)
cv2.imshow("outliers removed", outlier_removed)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)

#save the images

file_name=args.input_file.split('/')[-1]

cv2.imwrite(os.path.join(args.output_folder+'/',file_name),outlier_removed)
