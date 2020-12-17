import os
from collections import Counter

TOP = 20

data_path = '/netscratch/mmommert/smoke_emission/2020Neurips_dataset/images/training'
subsets = ['positive', 'negative']

all_images = []
for subset in subsets:
    image_path = os.path.join(data_path, subset)
    for (dirpath, dirname, filenames) in os.walk(image_path):
        for filename in filenames:
            if filename.endswith('.tif'):
                all_images.append(filename.split('_')[0])

counter = Counter(all_images)

print(counter.most_common(TOP))
