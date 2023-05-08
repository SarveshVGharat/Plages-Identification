import cv2
from segment import *
from configs import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Define a function to display the image and results
def display_results(img, output_img, corrected_area, calculated_area):
    # Display input image
    st.image(img, caption='Input Image')

    # Display output image
    st.image(output_img, caption='Output Image')

    # Display corrected and calculated areas
    results_dict = {'Corrected Area': [corrected_area], 'Calculated Area': [calculated_area]}
    st.table(results_dict)


# Define the first page of the app
def page1():
    st.header('Image Processing Playground')
    st.write('Upload an image and adjust the parameters to segment the image.')
    st.write('Input file name should be of the form CAK_YYYYMMDDTHHMMSS_Q1L1b800px.jpg where YYYYMMDDTHHMMSS is the date and time of the captured image in UTC+5:30.')

    # Upload the input image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Define the threshold slider
    thresh = st.slider('Binary Threshold', 0, 255, default_binary_thresh)

    # Define the area threshold slider
    area_thresh = st.slider('Upper Area Threshold', 0, 10000, default_area_thresh)

    lower_area_thresh = st.slider('Lower Area Threshold', 0, 500, default_lower_area_thresh)

    # Define the clip limit slider
    clip_limit = st.slider('Clip Limit', 0.0, 10.0, 2.0)

    # Check if an image is uploaded
    if uploaded_file is not None:

        file_name = uploaded_file.name 

        # Read the image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)

        # Call the algorithm function to segment the image
        corrected_area, calculated_area, output_image = algorithm(input_file = file_name, thresh = thresh,clip_limit=clip_limit, area_thresh=area_thresh, input_image=img, lower_area_thresh=lower_area_thresh, override=True)

        if corrected_area == -99:
            corrected_area = 'Unavailable'
        # Display the results
        display_results(img, output_image, corrected_area, calculated_area)

# Define the second page of the app
def page_x(cycle = None):
    st.header('Solar Cycle {} Analysis'.format(cycle))
    st.write('Two plots are displayed below with descriptions and a link to a Google Drive folder containing output images for solar cycle {} is present.'.format(cycle))
    table_id = project_id + '.' + 'cycle-' + str(cycle)

    # Plot 1
    df = get_dataframe(id = table_id)

    st.subheader('Plot 1')
    st.write('This plot shows the variation of calculated vs actual area of solar plages for images obtained from the Kodaikanal Solar Observatory during Solar Cycle {}'.format(cycle))
    fig = plot_time_series(df, return_fig=True)
    st.pyplot(fig)

    # Plot 2
    st.subheader('Plot 2')
    st.write('This plot shows the scatter plot between calculated and actual area of solar plages for images obtained from the Kodaikanal Solar Observatory during Solar Cycle {}'.format(cycle))
    fig2 = plot_scatter_plot(df, return_fig=True)
    st.pyplot(fig2)

    # Link to Google Drive folder
    st.write('Output images for solar cycle {} can be found at this link:'.format(cycle), 
             '[Google Drive folder](https://drive.google.com/drive/folders/1nMKew8hG8Eo6Ej1jSz7uSgyiW4C4AzGK?usp=sharing)')

# Define the app
def app():
    st.set_page_config(page_title="Image Processing App", page_icon=":camera:", layout="wide")

    # Create a menu with two options
    menu = [
        "Image Processing", 
        "Solar Cycle 22", 
        "Solar Cycle 21", 
        "Solar Cycle 20",
        "Solar Cycle 19", 
        "Solar Cycle 18",
        "Solar Cycle 17",
        "Solar Cycle 16"
    ]
    choice = st.sidebar.selectbox("Select an option", menu)

    # Show the appropriate page based on the user's choice
    if choice == "Image Processing":
        page1()
    elif choice == "Solar Cycle 22":
        page_x(cycle = 22)
    elif choice == "Solar Cycle 21":
        page_x(cycle = 21)
    elif choice == "Solar Cycle 20":
        page_x(cycle = 20)
    elif choice == "Solar Cycle 19":
        page_x(cycle = 19)
    elif choice == "Solar Cycle 18":
        page_x(cycle = 18)
    elif choice == "Solar Cycle 17":
        page_x(cycle = 17)
    elif choice == "Solar Cycle 16":
        page_x(cycle = 16)

if __name__ == '__main__':
    app()