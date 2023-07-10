from configs import *
from segment import *


def process_image(image, hyperparameters, image_name):
    corrected_area, calculated_area, segmented_image = algorithm(
        input_image=image,
        input_file=image_name,
        thresh=hyperparameters['threshold'],
        clip_limit=hyperparameters['clip_limit'],
        area_thresh=hyperparameters['upper_area_threshold'],
        lower_area_thresh=hyperparameters['lower_area_threshold'],
        override=True,
        get_segmented_image=True
    )

    return corrected_area, calculated_area, segmented_image

def check_duplicate_filename(file_path, filename):
    if os.path.isfile(file_path):
        existing_data = pd.read_csv(file_path)
        if filename in existing_data['Filename'].values:
            return True
    return False

def display_results(img, output_img, corrected_area, calculated_area):
    # Display input image
    st.image(img, caption='Input Image')

    # Display output image
    st.image(output_img, caption='Output Image')

    # Display corrected and calculated areas
    results_dict = {'Corrected Area': [corrected_area], 'Calculated Area': [calculated_area]}
    st.table(results_dict)

def save_results(cycle_id, image_name, image, segmented_image, corrected_area, calculated_area):
    output_folder = './annotation-results'
    input_path = output_folder + '-' + cycle_id + '/input/' + image_name
    output_path = output_folder + '-' + cycle_id + '/output/' + image_name
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(input_path, image)
    cv2.imwrite(output_path, segmented_image)

    data = {'Filename': [image_name], 'Corrected Area': [corrected_area], 'Calculated Area': [calculated_area]}
    df = pd.DataFrame(data)
    df.to_csv('logs.csv', mode='a', index=False)

def main():
    st.title("Image Segmentation Tool")
    cycle_id = st.sidebar.text_input("Solar Cycle ID")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    hyperparameters = {
            'threshold': st.slider("Threshold", 0, 255, default_binary_thresh),
            'clip_limit': st.slider("Clip Limit", 0.0, 10.0, default_clip_limit),
            'upper_area_threshold': st.slider("Upper Area Threshold", 0, 10000, default_area_thresh),
            'lower_area_threshold': st.slider("Lower Area Threshold", 0, 500, default_lower_area_thresh)
    }
    
    if uploaded_file is not None:
        
        image_name = uploaded_file.name 
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)
        
        if check_duplicate_filename('logs.csv', image_name):
            st.warning("Warning: File with the same name already exists in the CSV file.")
            return

        corrected_area, calculated_area, segmented_image = process_image(image, hyperparameters, image_name)
            
        if corrected_area == -99:
            corrected_area = "N/A"
            
        display_results(image, segmented_image, corrected_area, calculated_area)

        if st.button("Save Results"):
            save_results(cycle_id, image_name, image, segmented_image, corrected_area, calculated_area)
            st.success("Results saved successfully!")

if __name__ == "__main__":
    main()
