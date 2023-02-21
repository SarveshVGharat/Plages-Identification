import os
import cv2
import math
import tqdm
import numpy as np
import pandas as pd
import datetime
import multiprocessing
from google.cloud import bigquery
from circle_fit import *
from astropy.time import Time


credentials_path = './privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
df = pd.read_csv('./plage_areas.txt', header=None, skiprows=34, names=["Year", "Month", "Day", "Projected Area", "Corrected Area"], delim_whitespace = True)
table_id = 'custom-zone-377316.solar_plage_project.plage_stats'
images_dir = './images'
output_image_dir = './output_images'

def algorithm(input_file =  None, thresh = None, clip_limit = None, area_thresh = None):

    date_time = return_date_and_time(file = input_file)
    date, time = date_time.split(" ")
    corrected_area = get_corrected_area(date = date)
    if corrected_area == -99:
        return False, False, False, False
    
    input_image = cv2.imread(input_file,0)

    binary_threshold = thresh 

    disc_contour = return_solar_circumference_contour(image = input_image)

    disc_area = cv2.contourArea(disc_contour)

    circumference_removal_mask = np.zeros(input_image.shape, dtype="uint8")

    cv2.drawContours(circumference_removal_mask, disc_contour, -1, (255,255,255), thickness = 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    clahe = cv2.createCLAHE(clipLimit = clip_limit) #2
    clahe_image = clahe.apply(input_image)

    _, thresh_img = cv2.threshold(clahe_image, binary_threshold, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    thresh_img = cv2.erode(thresh_img,kernel,iterations=2)
    thresh_img = cv2.dilate(thresh_img,kernel,iterations=1)

    

    new_mask_M = np.zeros( input_image.shape, dtype="uint8" )

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            if circumference_removal_mask[i,j] == thresh_img[i,j]:
                thresh_img[i,j] = 0

    output_image = thresh_img

    disc_contour = np.squeeze(disc_contour)
    disc_contour = [list(ele) for ele in disc_contour]
    Xc, Yc, R, sigma = taubinSVD(disc_contour)
    Xc = np.ceil(Xc)
    Yc = np.ceil(Yc)
    Bo, Lo, P, Rad = calc_Bo_Lo_P_Rad(date_str = date_time)
    contours, h = cv2.findContours(output_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    area = 0.0

    for contour in contours:
        sigma_thetaIn = 0
        sigma_lIn = 0
        sigma_In =0

        mask = np.zeros(output_image.shape, dtype="uint8" )
        cv2.drawContours(mask, [contour],-1,(255,255,255),-1)
        candidate_points = return_non_zero_points(image = mask)
        for point in candidate_points:
            i = point[0]
            j = point[1]
            In = mask[i,j]
            x = j
            y = i

            r, theta_prime = convert_to_polar(x = x, y = y, Xc = Xc, Yc = Yc, R = R)
            T = Rad/15
            R_nought = Rad/(7*36)*(1e-12)*29.5953*np.cos(math.radians(math.degrees(np.arccos(-0.00629*T))/3+240))
            rho_prime = R_nought * r/R


            rho = math.degrees(np.arcsin(np.sin(math.radians(rho_prime))/np.sin(math.radians(R_nought)))) - rho_prime
            
            sin_theta = np.cos(math.radians(rho))*np.sin(math.radians(Bo)) + np.sin(math.radians(rho))*np.cos(math.radians(Bo))*np.sin(math.radians(theta_prime))
            cos_theta = np.sqrt(1-sin_theta**2)
            sin_l = np.cos(math.radians(theta_prime))*np.sin(math.radians(rho))/cos_theta

            theta = np.arcsin(sin_theta)
            l = np.arcsin(sin_l)

            sigma_thetaIn = sigma_thetaIn + theta*In
            sigma_lIn = sigma_lIn + l*In 
            sigma_In = sigma_In + In 

        theta_plage = sigma_thetaIn/sigma_In
        l_plage = sigma_lIn/sigma_In

        cos_delta = np.sin(math.radians(Bo))*np.sin(theta_plage)+np.cos(math.radians(Bo))*np.cos(theta_plage)*np.cos(l_plage)
        pixel_size = Rad/R
        pixel_area = pixel_size**2
        #area = area + (cv2.contourArea(contour)*pixel_area)/(2*np.pi*(Rad)**2 * cos_delta)
        contour_area = (cv2.contourArea(contour))/(cos_delta)
        disc_area_px = disc_area*pixel_area

        if contour_area < area_thresh:
            area = area + (contour_area*pixel_area)/disc_area_px
            new_mask_M = new_mask_M + mask
        
    output_image = new_mask_M

    #output_image = cv2.morphologyEx(output_image, cv2.MORPH_OPEN, kernel)

    return corrected_area, area, input_image, output_image

    

def return_solar_disc_radius(image = None):
        
    '''_, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)'''

    solar_disc = return_solar_disc(image = image)

    contour = return_solar_circumference_contour(image = solar_disc)

    ellipse = cv2.fitEllipse(contour)

    (x, y), (Ma, ma), angle = ellipse

    radius = (Ma + ma) / 4

    return radius

def return_solar_disc(image = None):

    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


def return_solar_circumference_contour(image = None):

    contours, h = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours[0]

def return_solar_centre(disc_image = None):
    Xc = 0
    Yc = 0
    L = 0
    for i in range(disc_image.shape[0]):
        for j in range(disc_image.shape[1]):
            if disc_image[i,j] > 0:
                Xc+=255*j
                Yc+=255*i 
                L+=255
    Xc = Xc/L
    Yc = Yc/L
    return Xc,Yc 

#def calculate_contour_area(contour = None):


def calculate_area(input_image = None, output_image =  None, date_time  = None):

    disc_contour = return_solar_circumference_contour(image = input_image)
    disc_area = cv2.contourArea(disc_contour)
    disc_contour = np.squeeze(disc_contour)
    disc_contour = [list(ele) for ele in disc_contour]
    Xc, Yc, R, sigma = taubinSVD(disc_contour)
    Xc = np.ceil(Xc)
    Yc = np.ceil(Yc)
    Bo, Lo, P, Rad = calc_Bo_Lo_P_Rad(date_str = date_time)
    contours, h = cv2.findContours(output_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    area = 0.0


    for contour in contours:
        sigma_thetaIn = 0
        sigma_lIn = 0
        sigma_In =0

        mask = np.zeros(output_image.shape, dtype="uint8" )
        cv2.drawContours(mask, [contour],-1,(255,255,255),-1)
        candidate_points = return_non_zero_points(image = mask)
        for point in candidate_points:
            i = point[0]
            j = point[1]
            In = mask[i,j]
            x = j
            y = i

            r, theta_prime = convert_to_polar(x = x, y = y, Xc = Xc, Yc = Yc, R = R)
            T = Rad/15
            R_nought = Rad/(7*36)*(1e-12)*29.5953*np.cos(math.radians(math.degrees(np.arccos(-0.00629*T))/3+240))
            rho_prime = R_nought * r/R


            rho = math.degrees(np.arcsin(np.sin(math.radians(rho_prime))/np.sin(math.radians(R_nought)))) - rho_prime
            
            sin_theta = np.cos(math.radians(rho))*np.sin(math.radians(Bo)) + np.sin(math.radians(rho))*np.cos(math.radians(Bo))*np.sin(math.radians(theta_prime))
            cos_theta = np.sqrt(1-sin_theta**2)
            sin_l = np.cos(math.radians(theta_prime))*np.sin(math.radians(rho))/cos_theta

            theta = np.arcsin(sin_theta)
            l = np.arcsin(sin_l)

            sigma_thetaIn = sigma_thetaIn + theta*In
            sigma_lIn = sigma_lIn + l*In 
            sigma_In = sigma_In + In 

        theta_plage = sigma_thetaIn/sigma_In
        l_plage = sigma_lIn/sigma_In

        cos_delta = np.sin(math.radians(Bo))*np.sin(theta_plage)+np.cos(math.radians(Bo))*np.cos(theta_plage)*np.cos(l_plage)
        pixel_size = Rad/R
        pixel_area = pixel_size**2
        #area = area + (cv2.contourArea(contour)*pixel_area)/(2*np.pi*(Rad)**2 * cos_delta)
        area = area + (cv2.contourArea(contour)*pixel_area)/(disc_area*pixel_area * cos_delta)
        
    return area
                        



    
    #area = area/(2*np.pi*R**2)
    
    #return area

def calc_Bo_Lo_P_Rad(date_str =  None):

    time = Time(date_str, format = 'iso', scale = 'utc')
    time.delta_ut1_utc = -19800

    time = Time(time.ut1.iso, format = 'iso', scale = 'utc')

    JD = time.jd
    
    T = (JD-2415020)/36525

    L_prime = 279.69668 + 36000.76892*T + 0.0003025*(T**2)

    g = 358.47583 + 35999.04975*T - 0.00015*(T**2) - 0.0000033*(T**3)

    ohm =  259.18 - 1934.142*T

    C1 = (1.91946 - 0.004789*T - 0.000014*(T**2))*np.sin(math.radians(g))

    C2 = (0.020094 - 0.0001*T)* np.sin(math.radians(2*g))

    C3 = 0.0001*np.sin(math.radians(3*g))

    C = C1+C2+C3

    Lambda_nought = L_prime + C

    Lambda_a = Lambda_nought - 0.00569 - 0.00479* np.sin(math.radians(ohm))

    #20230206063328

    phi = (360/25.38)*(JD - 2398220)

    if not(phi>=0 and phi<=360):

        while not(phi>=0 and phi<=360):

            if phi>360:
                phi = phi-360
            else:
                phi = phi+360


    #scale phi to zero to 360

    K = 74.3646 + 1.395833*T

    I = 7.25

    epsilon_nought = 23.452295 - 0.0130125*T - 0.00000164*(T**2) + 0.000000593*(T**3)

    epsilon = epsilon_nought + 0.00256*np.cos(math.radians(ohm))
    
    X = np.arctan(-1*np.cos(math.radians(Lambda_a))*np.tan(math.radians(epsilon)))

    Y = np.arctan(-1* np.cos(math.radians(Lambda_nought - K))*np.tan(math.radians(I)))

    P = math.degrees(X+Y)

    Bo = math.degrees(np.arcsin(np.sin(math.radians(Lambda_nought - K))*np.sin(math.radians(I))))

    M = 360 - phi

    Lo = math.degrees(np.arctan((np.sin(math.radians(K-Lambda_nought))*np.cos(math.radians(I)))/(-1* np.cos(math.radians(K-Lambda_nought))))) + M 

    R_prime = 1.00014 - 0.01671*np.cos(math.radians(g)) - 0.00014 * np.cos(math.radians(2*g))

    Rad = 0.2666/R_prime * 3600

    return Bo, Lo, P, Rad 

    #20230206070710

def convert_to_polar(x = None, y = None, Xc = None, Yc= None, R = None):

    x = x - Xc
    y = y - Yc

    theta_prime = math.degrees(np.arctan2(y,x))

    r = np.clip(np.sqrt(x**2+y**2), a_max = R-1, a_min = 0)

    return r, theta_prime

#20230206084229

def return_thresh(image = None):

    I =[]
    for x in image:
        for y in x:
            if y>0:
                I.append(y)
    
    u = np.mean(I)
    s = np.std(I)

    return int(u+1.75*s)

def return_non_zero_points(image = None):

    non_zero_indices = np.nonzero(image)
    non_zero_indices = zip(non_zero_indices[0], non_zero_indices[1])

    return non_zero_indices



def return_date_and_time(file = None):

    date_time = file.split("_")[1].replace("T", " ")
    dt = datetime.datetime.strptime(date_time, "%Y%m%d %H%M%S")
    formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_dt


def get_corrected_area(date = None):

    year, month, day = map(int, date.split("-"))

    row = df.loc[(df["Year"] == year) & (df["Month"] == month) & (df["Day"] == day)]

    corrected_area = row["Corrected Area"].values[0]

    return corrected_area

def create_table():
    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("date", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("time", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("corrected_area", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("calculated_area", "FLOAT", mode="NULLABLE")
    ]

    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table) 

def delete_table():

    client = bigquery.Client()
    client.delete_table(table_id, not_found_ok=True)


def append_to_table(date = None, time = None, corrected_area = None, calculated_area = None):

    client = bigquery.Client()

    rows_to_insert = [
        {u'date':date, u'time':time, u'corrected_area':corrected_area, u'calculated_area':calculated_area},
    ]

    client.insert_rows_json(table_id, rows_to_insert)

def pipeline(file = None):

    file_name = file.split('/')[-1]
    corrected_area, calculated_area, input_image, output_image = algorithm(input_file = file, thresh = 180, clip_limit=2, area_thresh = 1000)
    diff = corrected_area - calculated_area
    if diff>0.01:
        corrected_area, calculated_area, input_image, output_image = algorithm(input_file = file, thresh = 170, clip_limit=2, area_thresh = 1000)
    elif diff < -0.01 :
        corrected_area, calculated_area, input_image, output_image = algorithm(input_file = file, thresh = 190, clip_limit=2, area_thresh = 1000)
    formatted_dt = return_date_and_time(file)
    date, time = formatted_dt.split(" ") 
    if not(corrected_area == False): 
        append_to_table(date=date, time=time, corrected_area=corrected_area, calculated_area=calculated_area)
        cv2.imwrite(filename = output_image_dir + '/' + file_name, img = output_image)

def main():

    #delete_table()
    os.makedirs(output_image_dir, exist_ok=True)
    files = os.listdir(images_dir)
    input_files = [images_dir + '/'+ file for file in files]
    create_table()
    pool = multiprocessing.Pool(processes = 10)
    for _ in tqdm.tqdm(pool.imap_unordered(pipeline, input_files), total=len(input_files)):
        pass
    
if __name__ == '__main__':
    main()
