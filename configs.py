import os
import pandas as pd
credentials_path = './privatekey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
df = pd.read_csv('./plage_areas.txt', header=None, skiprows=34, names=["Year", "Month", "Day", "Projected Area", "Corrected Area"], delim_whitespace = True)
project_id = 'custom-zone-377316.solar_plage_project'
cycle_id = 21
images_dir_prefix = './images-cycle-'
output_image_dir_prefix = './output-images-cycle-'
table_id = project_id + '.' + 'cycle-' + str(cycle_id)
images_dir = images_dir_prefix + str(cycle_id)
output_image_dir = output_image_dir_prefix + str(cycle_id)
clip_limit_value = 2
area_thresh_value  = 1400
#1400 for 22