import lasio
import pandas as pd
import numpy as np
import os

def las_oth_fix(las_file):
    for curve in las_file.curves:
        men = curve.mnemonic
        des = curve.descr
        if 'oth' in men.lower():
            curve.mnemonic = des.split()[1]

def PreProcess(data_dir):
    wellheads_df = pd.DataFrame(columns=['filename', 'well_name', 'x', 'y', 'lat', 'lon'])
    merged_data_df = pd.DataFrame(columns=['filename', 'well_name', 'x', 'y', 'lat', 'lon'])
    i = 0
    for filename in os.listdir(data_dir):
        # if not filename.startswith('.'):
        if filename.lower().endswith(".las"):

            # Open las
            las_file = lasio.read(os.path.join(data_dir, filename))

            #check for oth error
            oth_check = [i for i in las_file.curves.keys() if 'oth' in i.lower()]
            if oth_check:
                #print(f"Fixing {filename}")
                las_oth_fix(las_file)

            well_name = las_file.well.WELL.value
            if 'X' in las_file.well.keys():
                if las_file.well.X.value == '':
                    x = np.nan
                else:
                    x = las_file.well.X.value
            else:
                x = np.nan
            if 'Y' in las_file.well.keys():
                if las_file.well.Y.value == '':
                    y = np.nan
                else:
                    y = las_file.well.Y.value
            else:
                y = np.nan
            if 'LATI' in las_file.well.keys(): 
                if las_file.well.LATI.value == '':
                    lat = np.nan
                else:
                    lat = las_file.well.LATI.value
            else:
                lat = np.nan
            if 'LONG' in las_file.well.keys():
                if las_file.well.LONG.value == '':
                    lon = np.nan
                else:
                    lon = las_file.well.LONG.value
            else:
                lon = np.nan
            logs = las_file.header['Curves'].keys()
            temp_df = pd.DataFrame({'filename':[filename], 'well_name':[well_name], 'x':[x], 'y':[y], 'lat':[lat], 'lon':[lon], 'logs':[logs]})
    #         if 'COMPOS' in filename:
    #             composite_df = pd.concat([composite_df,temp_df], axis=0, ignore_index=True, sort=False)
    #         elif 'INTERPRET' in filename:
    #             interprete_df = pd.concat([interprete_df,temp_df], axis=0, ignore_index=True, sort=False) 
            wellheads_df = pd.concat([wellheads_df,temp_df], axis=0, ignore_index=True, sort=False)
            
            temp_df = las_file.df() 
            temp_df.replace(-999.25,np.nan)
            #     df.dropna
            temp_df.insert(0,'DEPTH', temp_df.index)
            temp_df.insert(0,'lon', lon)
            temp_df.insert(0,'lat', lat)
            temp_df.insert(0,'y', y)
            temp_df.insert(0,'x', x)
            temp_df.insert(0,'well_name', well_name)
            temp_df.insert(0,'filename', filename)
            print(well_name)
            merged_data_df = pd.concat([merged_data_df,temp_df], axis=0, ignore_index=True, sort=False)
    merged_data_df.to_csv('mergeddf.csv', header = True)

def filterDataframe(csv_file):

    dataclean = pd.read_csv(csv_file)
    dataclean = dataclean[['filename', 'well_name','DEPTH','DT','GR','RDEP','RMED','RSHA','GR:1','GR:2','DT:1','DT:2']]

    dataclean["GR"] = np.where(dataclean['GR'].isna(), dataclean['GR:1'], dataclean['GR'])
    dataclean["GR"] = np.where(dataclean['GR'].isna(), dataclean['GR:2'], dataclean['GR'])

    dataclean["DT"] = np.where(dataclean['DT'].isna(), dataclean['DT:1'], dataclean['DT'])
    dataclean["DT"] = np.where(dataclean['DT'].isna(), dataclean['DT:2'], dataclean['DT'])

    dataclean["RES"] = np.where(dataclean['RDEP'].isna(), dataclean['RMED'], dataclean['RDEP'])
    
    ResultDF = dataclean[["filename", "well_name", "DEPTH", "GR", "DT", "RES"]].dropna()
    ResultDF.to_csv("resultDF_ALLLOGS.csv", header = True)

    return ResultDF

def formation_tops(filename):
    tops = pd.read_csv(filename)
    topdf = pd.DataFrame(tops)
    
    formations_dict = {k: f.groupby('MD')['Surface'].apply(list).to_dict()
    for k, f in topdf.groupby('Well identifier')}

    return formations_dict

def add_formation_name_to_df(depth, well_name):

    formations_depth = formations_dict[well_name].keys()   
    # Need to catch if we are at the last formation
    try:
        at_last_formation = False
        below = min([i for i in formations_depth if depth < i])
    except ValueError:
        at_last_formation = True

    # Need to catch if we are above the first listed formation
    try:
        above_first_formation = False
        above = max([i for i in formations_depth if depth > i])
    except:
        above_first_formation = True

    if above_first_formation:
        formation = ''

    else:# Check if the current depth matches an existing formation depth
        nearest_depth = min(formations_depth, key=lambda x:abs(x-depth))
        if depth == nearest_depth:
            formation = formations_dict[well_name][nearest_depth][0]

        else:
            if not at_last_formation:
                if depth >= above and depth <below:
                    formation = formations_dict[well_name][above][0]
            else:
                formation = formations_dict[well_name][above][0]
    return formation

formations_dict = formation_tops()
ResultDF = filterDataframe()
ResultDF = ResultDF[ResultDF["GR"] >= 0]
ResultDF['FORMATION'] = ResultDF.apply(lambda x: add_formation_name_to_df(x['DEPTH'], x['well_name']), axis=1)
ResultDF.to_csv('dfcleaned.csv', header=True)
print(ResultDF['FORMATION'].unique())
