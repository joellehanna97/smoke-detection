import pandas as pd
import json
import os
import cdsapi
import xarray as xr
import numpy as np

already_downloaded = True


def request_data(month):
    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-pressure-levels",
               {
                   'product_type': 'reanalysis',
                   'variable': [
                       'specific_humidity', 'temperature', 'u_component_of_wind',
                       'v_component_of_wind',
                   ],
                   'year': '2019',
                   'month': [
                       month,
                   ],
                   'day': [
                       '01', '02', '03',
                       '04', '05', '06',
                       '07', '08', '09',
                       '10', '11', '12',
                       '13', '14', '15',
                       '16', '17', '18',
                       '19', '20', '21',
                       '22', '23', '24',
                       '25', '26', '27',
                       '28', '29', '30',
                       '31',
                   ],
                   'time': [
                       '00:00', '01:00', '02:00',
                       '03:00', '04:00', '05:00',
                       '06:00', '07:00', '08:00',
                       '09:00', '10:00', '11:00',
                       '12:00', '13:00', '14:00',
                       '15:00', '16:00', '17:00',
                       '18:00', '19:00', '20:00',
                       '21:00', '22:00', '23:00',
                   ],
                   'pressure_level': '1000',
                   "format": "grib"}, "download.grib")

    ds = xr.open_dataset('download.grib', engine='cfgrib')
    ds_ar = ds.to_array()
    return ds_ar


def get_data(lat, lon, filename):
    latitude = np.linspace(90, -90, 721)
    longitude = np.linspace(0, 359.75, 1440)

    month = int(filename.split(' ')[0].split('-')[1])
    day = int(filename.split(' ')[0].split('-')[2])
    hour = int(filename.split(' ')[1].split(':')[0])
    time_idx = (day - 1) * 24 + hour

    ds_ar = np.load('weather_{}.npy'.format(month))
    lat_id = np.where(latitude == min(latitude, key=lambda x: abs(x - lat)))[0][0]
    lon_id = np.where(longitude == min(longitude, key=lambda x: abs(x - lon)))[0][0]

    humidity = ds_ar[0, time_idx, lat_id, lon_id]
    temperature = ds_ar[1, time_idx, lat_id, lon_id]
    u_wind = ds_ar[2, time_idx, lat_id, lon_id]
    v_wind = ds_ar[3, time_idx, lat_id, lon_id]

    return humidity, temperature, u_wind, v_wind


def clean_dir():
    files_in_directory = os.listdir('.')
    filtered_files = [file for file in files_in_directory if 'grib' in file]
    for file in filtered_files:
        path_to_file = os.path.join('.', file)
        os.remove(path_to_file)


with open('mapping.json') as json_file:
    dict_id_names = json.load(json_file)

new_dir = '2020Neurips_dataset/images_subset/training'
subsets = ['positive', 'negative']

df_info = pd.read_csv('power_info.csv')
df_labels = pd.DataFrame(columns=['filename', 'lat', 'lon', 'label', 'EIC', 'temp', 'humidity', 'wind-u',
                                  'wind-v', 'gen_output'])

# Load extra resources - keep only data for the year 2019.
facilities = pd.read_excel('resources/EPRTR_facilities.xlsx')

if not already_downloaded:
    for i in range(1, 13):
        ds = request_data('{:02d}'.format(i))
        np.save('weather_{}'.format(i), ds)
        clean_dir()

# Create the final dataset
for subset in subsets:
    image_path = os.path.join(new_dir, subset)
    for (dirpath, dirname, filenames) in os.walk(image_path):
        for filename in filenames:
            if filename.endswith('.tif'):
                key = filename.split('_')[0]
                lat = facilities[facilities.FacilityID == int(key)]['Latitude'].values[0]
                lon = facilities[facilities.FacilityID == int(key)]['Longitude'].values[0]
                time = filename.split('_', 1)[1].replace('T', ' ')

                hum, temp, windu, windv = get_data(lat, lon, time)

                gen_output = sum(df_info[((df_info.DateTime.str.startswith(time[:-17])) & (df_info.GenerationUnitEIC.isin(dict_id_names[key])))]['ActualGenerationOutput'].tolist())
                df_labels = df_labels.append({'label': subset, 'lat': lat, 'lon': lon,
                                              'gen_output': gen_output, 'filename': filename,
                                              'EIC': dict_id_names[key], 'temp': temp, 'humidity': hum,
                                              'wind-u': windu, 'wind-v': windv}, ignore_index=True)

df_labels.to_csv('labels.csv')
