import lasio
import pandas as pd
import numpy as np
import os
from pycaret.classification import *

class WellData():
    def __init__(self, path_to_las_files, feature_columns=None, path_to_formation_top=None):
        self.feature = feature_columns 
        self.formation_path = path_to_formation_top 
        self.merge_df = pd.DataFrame(columns=['filename', 'well_name', 'x', 'y', 'lat', 'lon'])
        if os.path.isdir(path_to_las_files):
            las_files = os.listdir(path_to_las_files)
        else:
            las_files = [path_to_las_files]
        for filename in las_files:
            if filename.lower().endswith(".las"):
                # Open las
                if os.path.isdir(path_to_las_files):
                    las_file = lasio.read(os.path.join(path_to_las_files, filename))
                else:
                    las_file = lasio.read(path_to_las_files)
                
                #check for oth error
                oth_check = [i for i in las_file.curves.keys() if 'oth' in i.lower()]
                if oth_check:
                    for curve in las_file.curves:
                        men = curve.mnemonic
                        des = curve.descr
                        if 'oth' in men.lower():
                            curve.mnemonic = des.split()[1]

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
                temp_df = las_file.df() 
                temp_df.replace(-999.25,np.nan)
                temp_df['DEPTH'] = temp_df.index
                temp_df['lon'] = lon
                temp_df['lat'] = lat
                temp_df['y'] = y
                temp_df['x'] = x
                temp_df['well_name'] = well_name
                temp_df['filename'] = filename
                self.merge_df = pd.concat([self.merge_df,temp_df], axis=0, ignore_index=True, sort=False)

    def add_formation_name_to_df(self, depth, well_name):
        topdf = pd.read_csv(self.formation_path)
        
        formations_dict = {k: f.groupby('MD')['Surface'].apply(list).to_dict() for k, f in topdf.groupby('Well identifier')}

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

    def process_data(self):
        filtered_columns = ["filename", "well_name", "DEPTH"] + self.feature
        filtered_df = self.merge_df[filtered_columns].dropna()
        if self.formation_path:
            filtered_df['FORMATION'] = filtered_df.apply(lambda x: self.add_formation_name_to_df(x['DEPTH'], x['well_name']), axis=1)
        return filtered_df

class Trainer():
    def __init__(self, session_id=123):
        self.session_id = session_id
        self.tuned_model = None

    def train(self,train_df,models=None):
        exp1 = setup(data=train_df, target="FORMATION", session_id=self.session_id)
        if len(models) == 1:
            best = create_model(models[0])
        else:
            best = compare_models(include=models)
        self.tuned_model = tune_model(best)
        return self.tuned_model

    def visualize(self, plot_type):
        if self.tuned_model:
            plot_model(self.tuned_model, plot =plot_type)
        else:
            print("Please load or train a model first")

    def save(self, filename):
        if self.tuned_model:
            save_model(self.tuned_model, filename)
        else:
            print("Please load or train a model first")

    def load(self, filename):
        self.tuned_model = load_model(filename)
        return self.tuned_model

    def predict(self,test_df):
        if self.tuned_model:
            yhat = predict_model(self.tuned_model, data = test_df)['Label']
            message = "Done prediction"
            return message, yhat
        else:
            message = "Please load or train a model first"
            return message, None