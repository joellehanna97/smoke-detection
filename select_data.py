import json
import os
from shutil import copyfile

with open('resources/mapping.json') as json_file:
    dict_id = json.load(json_file)


selected_ids = list(dict_id.keys())

new_dir = '/netscratch/jhanna/images_subset/training'
data_path = '/netscratch/mmommert/smoke_emission/2020Neurips_dataset/images/training'

if not os.path.exists(new_dir):
    os.makedirs(new_dir + '/positive')
    os.makedirs(new_dir + '/negative')

subsets = ['positive', 'negative']

all_images = []
for subset in subsets:
    image_path = os.path.join(data_path, subset)
    for (dirpath, dirname, filenames) in os.walk(image_path):
        for filename in filenames:
            if filename.endswith('.tif') and (filename.split('_')[0] in selected_ids):
                # copyfile(os.path.join(image_path, filename), os.path.join(new_dir, subset, filename))
                copyfile(os.path.join(image_path, filename), os.path.join(new_dir, subset, filename.replace(':', '-')))
