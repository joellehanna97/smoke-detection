import pandas as pd
import chardet
import json

with open('mapping.json') as json_file:
    dict_id = json.load(json_file)

selected_names = [item for sublist in list(dict_id.values()) for item in sublist]

csv_path = '/media/data/entsoe/2019_12_ActualGenerationOutputPerUnit.csv'
with open(csv_path, 'rb') as f:
    result = chardet.detect(f.read())

df_full = pd.DataFrame(columns=['Year', 'Month', 'Day', 'DateTime', 'ResolutionCode', 'AreaCode',
                                'AreaTypeCode', 'AreaName', 'MapCode', 'GenerationUnitEIC',
                                'PowerSystemResourceName', 'ProductionTypeName',
                                'ActualGenerationOutput', 'ActualConsumption', 'InstalledGenCapacity',
                                'UpdateTime'])

for month in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
    csv_path = '/media/data/entsoe/2019_{}_ActualGenerationOutputPerUnit.csv'.format(month)
    df_gen = pd.read_csv(csv_path, sep='\t', encoding=result['encoding'], dtype={"Year": int})
    df_gen = df_gen[df_gen.ProductionTypeName.str.startswith('Fossil')]  # Keep only fossil production
    df_gen = df_gen[df_gen.GenerationUnitEIC.isin(selected_names)]
    df_full = df_full.append(df_gen, ignore_index=True)

df_full.to_csv('power_info.csv')
