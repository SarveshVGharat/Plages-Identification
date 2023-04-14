import cv2
import math
import tqdm
import numpy as np
import datetime
import random
import multiprocessing
from configs import *
import matplotlib.pyplot as plt
from google.cloud.exceptions import NotFound
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from google.cloud import bigquery
from circle_fit import *
from astropy.time import Time

#try keeping lower limit as well
#add functionality to do images in bulk
#take a look at those papers mentioned by reviewer 2
def algorithm(input_file =  None, thresh = None, clip_limit = None, area_thresh = None, input_image = None):

    date_time = return_date_and_time(file = input_file)
    date, time = date_time.split(" ")
    corrected_area = get_corrected_area(date = date)
    if corrected_area == -99:
        return False,False,False
    if input_image is None:
        input_image = cv2.imread(input_file,0)
    else:
        input_image = input_image

    binary_threshold = thresh 

    disc_contour = return_solar_circumference_contour(image = input_image)

    disc_area = cv2.contourArea(disc_contour)

    circumference_removal_mask = np.zeros(input_image.shape, dtype="uint8")

    cv2.drawContours(circumference_removal_mask, disc_contour, -1, (255,255,255), thickness = 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    clahe = cv2.createCLAHE(clipLimit = clip_limit) #2
    clahe_image = clahe.apply(input_image)

    thresh_val, thresh_img = cv2.threshold(clahe_image, binary_threshold, 255, cv2.THRESH_BINARY)

    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    thresh_img = cv2.erode(thresh_img,kernel,iterations=2)
    thresh_img = cv2.dilate(thresh_img,kernel,iterations=1)
    

    new_mask_M = np.zeros( input_image.shape, dtype="uint8" )


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

        if contour_area < area_thresh:#and contour_area > 30:
            area = area + (contour_area*pixel_area)/disc_area_px
            new_mask_M = new_mask_M + mask
        
    output_image = new_mask_M
    contours, h = cv2.findContours(output_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(input_image_rgb, contours, -1, (0, 0, 255), 2)


    return corrected_area, area, input_image_rgb

def return_solar_circumference_contour(image = None):

    contours, h = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours[0]


def get_dataframe(id = None):
    client = bigquery.Client()

    # Set up the query
    query = """
        WITH base_query AS (
    SELECT 
        date,
        time,
        corrected_area,
        calculated_area
    FROM
        `{}`
    )
    SELECT 
        date, 
        STRING_AGG(time, ",") AS time, 
        AVG(corrected_area) AS corrected_area, 
        AVG(calculated_area) AS calculated_area 
    FROM 
        base_query 
    GROUP BY 
        date 
    ORDER BY 
        date ASC;
    """.format(id)


    query_job = client.query(query)

    dataframe = (
            query_job
            .result()
            .to_dataframe()
        )
    
    return dataframe

def plot_time_series(df = None, return_fig = False):
    rolling_window_size = 30
    df['corrected_area_rolling_mean'] = df['corrected_area'].rolling(rolling_window_size).mean()
    df['calculated_area_rolling_mean'] = df['calculated_area'].rolling(rolling_window_size).mean()
    fig = plt.figure(figsize=(8,6))
    plt.plot(
        df['corrected_area_rolling_mean'], 
        label='Corrected Area {} point Rolling Mean (Chatzistergos et al., 2020)'.format(rolling_window_size)
    )
    plt.plot(
        df['calculated_area_rolling_mean'], 
        label='Calculated Area {} point Rolling Mean (Our approach)'.format(rolling_window_size)
    )
    x_axis_ticks = np.arange(100, len(df), 150)
    x_axis_labels = df["date"].iloc[x_axis_ticks]
    plt.xticks(x_axis_ticks, x_axis_labels)
    plt.xlabel('date(dd-mm-yyyy)')
    plt.ylabel('plage area(disc fraction)')
    plt.legend()
    if return_fig:
        return fig
    plt.show()

def plot_scatter_plot(df = None, return_fig = False):
    fig = plt.figure(figsize=(8,6))
    X = df['corrected_area'].values.reshape(-1, 1)
    Y = df['calculated_area'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, Y)
    Y_pred = reg.predict(X)

    # Calculate the R^2 score
    r2 = r2_score(Y, Y_pred)

    # Calculate the correlation coefficient
    corr = np.corrcoef(X.reshape(-1), Y.reshape(-1))[0, 1]

    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 0.1])
    # Plot the scatter plot
    plt.scatter(X, Y)

    # Plot the regression line
    plt.plot(X, Y_pred, color='red')

    # Add the R^2 value and the correlation coefficient to the plot
    plt.text(0.08, 0.015, 'R^2 = {:.2f}'.format(r2), fontsize=10)
    plt.text(0.08, 0.01, 'r = {:.2f}'.format(corr), fontsize=10)

    # Set the x and y axis labels
    plt.xlabel("Corrected Area (Chatzistergos et al., 2020)")
    plt.ylabel("Calculated Area (Our approach)")
    if return_fig:
        return fig
    plt.show()


                        

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

def create_table(id = None):
    client = bigquery.Client()

    schema = [
        bigquery.SchemaField("date", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("time", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("corrected_area", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("calculated_area", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("thresh_manual", "INTEGER", mode="NULLABLE")
    ]

    table = bigquery.Table(id, schema=schema)
    table = client.create_table(table) 

def delete_table(id = None):

    client = bigquery.Client()
    client.delete_table(id, not_found_ok=True)

def delete_table_rows(id = None):

    client = bigquery.Client()
    query = """
        DELETE FROM `{}` WHERE true;
    """.format(id)
    query_job = client.query(query)

def append_to_table(id = None, date = None, time = None, corrected_area = None, calculated_area = None, thresh_manual = None):

    client = bigquery.Client()

    rows_to_insert = [
        {u'date':date, u'time':time, u'corrected_area':corrected_area, u'calculated_area':calculated_area, u'thresh_manual':thresh_manual },
    ]

    client.insert_rows_json(id, rows_to_insert)

def pipeline(file = None):

    file_name = file.split('/')[-1]
    thresh_list = [170, 180, 190, 200, 220]
    calculated_area = None
    corrected_area = None
    thresh_manual = None
    diff = None
    for thresh in thresh_list:
        corrected_area, calculated_area_temp, output_image_temp = algorithm(
            input_file = file, 
            thresh = thresh, 
            clip_limit=clip_limit_value, 
            area_thresh = 
            area_thresh_value
        )
        if not(corrected_area == False):
            thresh_manual = thresh
            if calculated_area is None:
                calculated_area = calculated_area_temp
                output_image = output_image_temp
                diff = np.abs(corrected_area - calculated_area)
            else:
                diff_temp = np.abs(corrected_area - calculated_area_temp)
                if diff_temp < diff:
                    calculated_area = calculated_area_temp
                    output_image = output_image_temp
                    diff = diff_temp
        else:
            break
    
    formatted_dt = return_date_and_time(file)
    date, time = formatted_dt.split(" ") 
    if not(corrected_area == False): 
        append_to_table(
            id = table_id,
            date=date, 
            time=time, 
            corrected_area=corrected_area, 
            calculated_area=calculated_area, 
            thresh_manual=thresh_manual
        )
        cv2.imwrite(filename = output_image_dir + '/' + file_name, img = output_image)


def main():

    client = bigquery.Client()
    try:
        client.get_table(table_id)
        print("Table {} already exists.".format(table_id))
        delete_table_rows(id = table_id)
    except NotFound:
        create_table(id = table_id)
    os.makedirs(output_image_dir, exist_ok=True)
    files = os.listdir(images_dir)
    input_files = [images_dir + '/'+ file for file in files]
    pool = multiprocessing.Pool(processes = 10)
    for _ in tqdm.tqdm(pool.imap_unordered(pipeline, input_files), total=len(input_files)):
        pass
    df = get_dataframe(id = table_id)
    plot_time_series(df = df)
    plot_scatter_plot(df = df)
    
    
if __name__ == '__main__':
    main()
