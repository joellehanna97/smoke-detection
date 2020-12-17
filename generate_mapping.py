import numpy as np
import pandas as pd
import chardet
import json


top_facilities = ['123521', '25075', '8618', '7350', '14276', '8093', '159137', '8243', '191317', '67', '115', '7514', '120277', '14254', '5038', '14255', '193180', '15873','177553', '15823']

top = pd.read_excel('/media/data/entsoe/EPRTR_facilities.xlsx')

# Keep only top facilties
top = top[top.FacilityID.isin(top_facilities)]

# Load extra resources to help the mapping
oppd_links = pd.read_csv('datasets/jrc_ppdb_open/JRC_OPEN_LINKAGES.csv')
oppd_units = pd.read_csv('datasets/jrc_ppdb_open/JRC_OPEN_UNITS.csv')
oppd = pd.merge(oppd_links, oppd_units, how='inner', on='eic_p')


csv_path = '/media/data/entsoe/2019_12_ActualGenerationOutputPerUnit.csv'
with open(csv_path, 'rb') as f:
    result = chardet.detect(f.read())

countries = ['ES', 'IT', 'GR', 'BG', 'RO']

df_full = pd.DataFrame(columns=['Year', 'Month', 'Day', 'DateTime', 'ResolutionCode', 'AreaCode',
                                'AreaTypeCode', 'AreaName', 'MapCode', 'GenerationUnitEIC',
                                'PowerSystemResourceName', 'ProductionTypeName',
                                'ActualGenerationOutput', 'ActualConsumption', 'InstalledGenCapacity',
                                'UpdateTime'])

for month in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
    csv_path = '/media/data/entsoe/2019_{}_ActualGenerationOutputPerUnit.csv'.format(month)
    df_gen = pd.read_csv(csv_path, sep='\t', encoding=result['encoding'], dtype={"Year": int})
    df_gen = df_gen[df_gen.ProductionTypeName.str.startswith('Fossil')]
    df_full = df_full.append(df_gen, ignore_index=True)

country_filter = df_full[df_full['MapCode'].isin(countries)]
mapping = {}
for _, row_s in country_filter.iterrows():
    selected = oppd[oppd.eic_g_x == row_s.GenerationUnitEIC]

    if len(selected) > 1:
        selected = selected.iloc[0]
        lat = selected.lat
        lon = selected.lon
    elif len(selected) == 1:
        lat = selected.lat.values[0]
        lon = selected.lon.values[0]
    else:
        pass

    for _, row_t in top.iterrows():
        if (np.abs(lat - row_t.Latitude) <= 0.005) and (np.abs(lon - row_t.Longitude) <= 0.005):
            mapping[row_t.FacilityID] = list(set(mapping.get(row_t.FacilityID, []) + [row_s.GenerationUnitEIC]))

json = json.dumps(mapping)
f = open("mapping.json", "w")
f.write(json)
f.close()
