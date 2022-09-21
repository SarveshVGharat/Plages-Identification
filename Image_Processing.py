import cv2
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--input_path', type=str, help='input path')
parser.add_argument('--output_path', type=str, help='output path')

args = parser.parse_args()

image_list = os.listdir(args.input_path)

for each_img in tqdm(image_list):
    # Reading the image from the present directory
    
    #image = cv2.imread("C:/Users/BHASKAR BOSE/Desktop/ground_truth/gt/CAK_19990910T024000_Q2L1b800px.jpg")
    image = cv2.imread(os.path.join(args.input_path,each_img))
    
    canvas=np.zeros(image.shape,np.uint8)
    imag=np.zeros(image.shape,np.uint8)
    # The initial processing of the image
    image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 3)
    final_img = clahe.apply(image_bw)

    final_img=cv2.erode(final_img,None,iterations=2)
    final_img=cv2.dilate(final_img,None,iterations=1)
    
    # Ordinary thresholding the same image
    _, ordinary_img = cv2.threshold(final_img, 180, 255, cv2.THRESH_BINARY)

    ordinary_img=cv2.erode(ordinary_img,None,iterations=1)
    ordinary_img=cv2.dilate(ordinary_img,None,iterations=1)

    li=[]
    canvases=[]
    contours_draw,hierarchy=cv2.findContours(ordinary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours_draw)):
        canvas=np.zeros(image.shape[0:2],np.uint8)
        cv2.drawContours(canvas,[contours_draw[i]],-1,(255,255,255),-1)
        li.append(canvas.sum()/255)
        canvases.append(canvas)

    #cv2.drawContours(imag, [contours_draw[13]], -1, (255, 255, 255), -1)
    mean=np.mean(li)
    std=np.std(li)
    thr=mean+3*std
    indexes=[]
    
    # Outlier detection
    for i in range(len(li)):
        if(li[i]>thr or li[i]>2000):
            indexes.append(i)
    im=ordinary_img
    for i in indexes:
        im=cv2.bitwise_xor(canvases[i],im)
    im=cv2.erode(im,None,iterations=1)
    
    edge_eliminator=np.zeros([800,800,3],np.uint8)
    edge_eliminator=cv2.circle(canvas,(400,400),200,(255,255,255),-1)
    im=cv2.bitwise_and(edge_eliminator,im)
    #im=cv2.dilate(im,None,iterations=1)
    
    cv2.imwrite(args.output_path + '/' + each_img.split('.')[0] +".png",im)
