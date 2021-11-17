import os
import numpy as np
import pandas as pd

input_folder = "input_data"
output_folder = "cleaned_data"


# Read all .csv files detected
def read_files(input_folder): 
    array_data = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"): 
            array_data.append(pd.read_csv(input_folder + "/" + filename))
            #print(input_folder + "/" + filename)
    return array_data

# Read UserX inputs
def read_user_data(input_folder):
    left_hand_data = read_files(input_folder + "/left-hand")
    right_hand_data = read_files(input_folder + "/right-hand")
    return left_hand_data, right_hand_data

'''
The folder structure needs to be similar to this
User1
    left-hand
        files
    right-hand
        files
User2
    left-hand
        files
    right-hand
        files
'''
def read_input_folder(input_folder):
    prefix = "User"
    array_user_data = []
    # Loop through all UserX folders 
    for filename in os.listdir(input_folder):
        if filename.startswith(prefix): 
            array_user_data.append(read_user_data(input_folder + "/" + filename))
    return array_user_data

def clean_data(array_data):
    print("Cleaning input data")
    # TODO: clean up the input data

def export_cleaned_data(output_folder):
    print("Exporting cleaned data to ",output_folder)
    # TODO: export cleaned data to output_folder 


def process_inputs():
    array_data = read_input_folder(input_folder)
    #print(array_data) # array_data[x][0] to access left-hand data of UserX
    cleaned_array_data = clean_data(array_data)
    export_cleaned_data(output_folder)
