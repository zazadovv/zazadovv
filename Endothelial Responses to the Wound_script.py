import os, glob
import pandas as pd
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow


#Make a GUI interface to choose directory where all csvs are stored
app = QApplication(sys.argv)

window = QMainWindow()
folderpath = QFileDialog.getExistingDirectory(window, 'Select Folders with all CSVs')
app.quit()

import re


def extract_second_e_f_values(filename):
    """Extract the second E and F values from the filename."""
    e_matches = re.findall(r'E(\d+)', filename)
    f_matches = re.findall(r'F(\d+-?\d*)', filename)

    # We expect at least two matches for both E and F values
    if len(e_matches) < 1 and len(f_matches) < 1:
        raise ValueError("Filename does not contain two E or F values.")
    
    if len(e_matches) < 2: 
      e_value = int(e_matches[0])

    else:
      e_value = int(e_matches[1])
    
    if len(f_matches) < 2: 
      f_value = int(f_matches[0])
    
    else:
      f_value = int(f_matches[1])  # Get the second F value

    return e_value, f_value


def sort_filenames(filenames):
    """Sort the list of filenames based on the second E and F values."""
    return sorted(filenames, key=lambda x: extract_second_e_f_values(x))


#Combine multiple csvs with the same keyword (e.g. Vessel) into one df
def df_combine_keyword(directory, key,Normalize = False):
    os.chdir(directory)
    df_filenames_key_list = []
    df_file_list = []
    df_filenames = glob.glob('*.csv')

    if "Master Analysis Sheet.csv" in df_filenames:
      df_filenames.remove("Master Analysis Sheet.csv")

    df_filenames_sort = sort_filenames(df_filenames)


    additional_dict = {}

    for df in df_filenames_sort:

        if key in df or key.lower() in df:
            df_filenames_key_list.append(df)


    for n in range(len(df_filenames_key_list)):
        df_name = df_filenames_key_list[n]

        df = pd.read_csv(df_name, index_col=0)
        df.rename(columns={df.columns[0]: n}, inplace=True)
        
        if Normalize == True:
           mean_val = df[n][0:10].mean()
           df[n] = df[n]/mean_val
          

        additional_dict['ID'] = "{}/{}".format(directory, df_name)
        additional_dict['Dextran@'] = key
        additional_dict['Genotype'] = n

        additional_pd = pd.DataFrame([additional_dict])
        additional_pd_transpose = additional_pd.transpose()
        additional_pd_transpose.rename(columns={additional_pd_transpose.columns[0]: n}, inplace=True)

        df_append = pd.concat([additional_pd_transpose, df],axis = 0)
        df_file_list.append(df_append)

    df_merge = pd.concat(df_file_list, axis=1)

    return df_merge


Wound_df_unnormalize = df_combine_keyword(folderpath,'Wound')
Vessel_df_unnormalize = df_combine_keyword(folderpath,'Vessel')
Bleed_df_unnormalize = df_combine_keyword(folderpath,'Bleedout')

Wound_df_normalize = df_combine_keyword(folderpath,'Wound',Normalize=True)
Vessel_df_normalize = df_combine_keyword(folderpath,'Vessel',Normalize=True)

try:
  Dilation_df_unnormalize = df_combine_keyword(folderpath,'Dilation')
  Dilation_df_normalize = df_combine_keyword(folderpath,'Dilation',Normalize=True)
  print("Dilation Files included")

  #Combine Wound, Vessel and Bleed dataframes into 1 dataframe, if there are missing columns, fill with 0.
  Master_Sheet = pd.concat([Wound_df_unnormalize,Vessel_df_unnormalize,Bleed_df_unnormalize,Dilation_df_unnormalize, Wound_df_normalize,
  Vessel_df_normalize, Dilation_df_normalize],axis = 0).fillna(0)

except:
  #Combine Wound, Vessel and Bleed dataframes into 1 dataframe, if there are missing columns, fill with 0.
  Master_Sheet = pd.concat([Wound_df_unnormalize,Vessel_df_unnormalize,Bleed_df_unnormalize, Wound_df_normalize,
  Vessel_df_normalize],axis = 0).fillna(0)
  print("Dilation Files not included")



#Save in the same directory as your choosen directory
save_folderpath = "{}{}".format(folderpath, '/Master Analysis Sheet.csv')

Master_Sheet.to_csv(save_folderpath,header=False)