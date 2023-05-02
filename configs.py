import os
from google.cloud import bigquery
import streamlit as st
from google.oauth2 import service_account
import pandas as pd
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials = credentials)
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(st.secrets["gcp_service_account"])
df = pd.read_csv('./plage_areas.txt', header=None, skiprows=34, names=["Year", "Month", "Day", "Projected Area", "Corrected Area"], delim_whitespace = True)
project_id = st.secrets["gcp_service_account"]["project_id"]+'.'+st.secrets["database_id"]["db_name"]
cycle_id = 16
images_dir_prefix = './images-cycle-'
output_image_dir_prefix = './output-images-cycle-'
table_id = project_id + '.' + 'cycle-' + str(cycle_id)
images_dir = images_dir_prefix + str(cycle_id)
output_image_dir = output_image_dir_prefix + str(cycle_id)
clip_limit_value = 2
area_thresh_dict = {
    22: 5000,
    21: 750,
    20: 1750,
    19: 3000,
    18: 1000,
    17: 5000,
    16: 6000
}
area_thresh_value  = area_thresh_dict[cycle_id]
lower_area_thresh_value = None