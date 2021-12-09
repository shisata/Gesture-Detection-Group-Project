import os
import numpy as np
import pandas as pd

'''
The folder structure needs to be similar to this
User1
    left-hand
        <.csv files>
    right-hand
        <.csv files>
    age.txt
    dominant_hand.txt
User2
    left-hand
        <.csv files>
    right-hand
        <.csv files>
    age.txt
    dominant_hand.txt
'''

input_folder = "input_data"
output_folder = "cleaned_data"
output_file = "result.csv"
user_folder_prefix = "user"

# Form master dataframe
column_names = ['user', 'age', 'dominant_hand','shape', 'hand_used', 'acc_x-std','acc_y-std', 'acc_z-std', 'g-force_x-std', 'g-force_y-std', 'g-force_z-std', 'acc_x-peaks', 'acc_y-peaks', 'acc_z-peaks', 'g-force_x-peaks', 'g-force_y-peaks', 'g-force_z-peaks']
master = pd.DataFrame(columns=column_names)

# array of cols in the data files which hold data which will be manipulated
data_cols=['ax','ay','az','gFx','gFy','gFz']

# Clean extraced data
def clean_data(file, data, user, age, hand, dominant_hand):
    # Array containing standard deviations of columns
    data_std = data.std()
    data_mean = data.mean()
    peak_list = list()

    # Counting peaks
    for x in data_cols:
        upper = (data_mean[x] + data_std[x] * 1.28)
        lower = (data_mean[x] - data_std[x] * 1.28)

        # Removing all the noise, only want highest peaks and lowest valleys
        peaks = data[(data[x] > upper) | (data[x] < lower)]
        peaks = peaks.reset_index()

        # Making counter to find non-contiguous time blocks, which indicate a new peak or valley
        peaks['counter'] = (peaks['index'].shift(-1)-peaks['index'] > 1)
        peak_list.append(peaks['counter'].sum())

    # Insert dataset results as row in master dataframe
    # row = [user, age, 'R', file[5], hand[0].capitalize(), data_std[4], data_std[5], data_std[6], data_std[1], data_std[2], data_std[3], peak_list[0], peak_list[1], peak_list[2], peak_list[3], peak_list[4], peak_list[5] ]
    shape = file[5]
    row = [
        user, 
        age, 
        dominant_hand, 
        shape, 
        hand, 
        data_std[4], 
        data_std[5], 
        data_std[6], 
        data_std[1], 
        data_std[2], 
        data_std[3], 
        peak_list[0], 
        peak_list[1], 
        peak_list[2], 
        peak_list[3], 
        peak_list[4], 
        peak_list[5] 
    ]
    master.loc[len(master)] = row


# Read all .csv files detected and process extracted data
def process_files(input_folder, user, age, hand, dominant_hand): 
    for file in os.listdir(input_folder):
        # Check if is .csv file
        if file.endswith(".csv"): 
            # Getting data from individual csv file and cleaning
            data = pd.read_csv(input_folder + "/" + file)
            clean_data(file, data, user, age, hand, dominant_hand)

# Loop through user folder to find specific file and read the first line from it
def read_first_line(input_folder, filename):
    for file in os.listdir(input_folder):
        if file == filename:
            with open(input_folder + '/' + file) as f:
                return f.readline()
    return None

# Read files in userX folder
def read_user_input(input_folder, user):
    age = read_first_line(input_folder, 'age.txt')
    dominant_hand = read_first_line(input_folder, 'dominant_hand.txt')
    left_hand = 'L'
    right_hand = 'R'

    # In case read data from age or dominant_hand is None then it should be adjusted to default value
    if (age == None):
        age = '0'
    if (dominant_hand == None):
        dominant_hand = right_hand

    left_hand_data = process_files(input_folder + "/left-hand", user, age, left_hand, dominant_hand) 
    right_hand_data = process_files(input_folder + "/right-hand", user, age, right_hand, dominant_hand)

# Read all input files from input_folder and process them, will be stored in master
def read_and_process_input():
    # Loop through all UserX folders 
    print("Reading and processing data from: " + input_folder)
    for user in os.listdir(input_folder):
        if user.startswith(user_folder_prefix): 
            read_user_input(input_folder + "/" + user, user)


# Export dataframe from master to csv file
def export_cleaned_data():
    print("Exporting data to " + output_folder)
    master.to_csv(output_folder + '/' + output_file)
    print("Done!! Now you can read processed data from " + output_folder + ", the file will be: " + output_file)

# Main function for this file, should be called in run_program.py
def process_inputs():
    read_and_process_input()
    export_cleaned_data()


####### Legacy Code #######

# def process_inputs():
#     # Form master dataframe
#     column_names = ['user', 'age', 'dominant_hand','shape', 'hand_used', 'acc_x-std','acc_y-std', 'acc_z-std', 'g-force_x-std', 'g-force_y-std', 'g-force_z-std', 'acc_x-peaks', 'acc_y-peaks', 'acc_z-peaks', 'g-force_x-peaks', 'g-force_y-peaks', 'g-force_z-peaks']
#     master = pd.DataFrame(columns=column_names)

#     # array of cols in the data files which hold data which will be manipulated
#     data_cols=['ax','ay','az','gFx','gFy','gFz']

#     # Usings loops to only go through the folder structure
#     for user in os.listdir(input_folder):
#         if user == 'user1':
#             age = 22
#         elif user == 'user2':
#             age = 56
#         elif user == 'user3':
#             age = 23
#         else:
#             age = 19

#         for hand in os.listdir(input_folder + "/" + user):
#             # Case to ignore git files
#             if hand.startswith('.'):
#                 break
#             for file in os.listdir(input_folder + "/" + user + "/" + hand):
#                 # Case to ignore git files
#                 if file.startswith('.'):
#                     break
#                 # Getting data from individual csv file and cleaning
#                 data = pd.read_csv(input_folder + "/" + user + "/" + hand + "/" + file)

#                 # Array containing standard deviations of columns
#                  data_std = data.std()
#                 peak_list = list()

#                 # Counting peaks
#                 for x in data_cols:
#                     upper = data_std[x] * 1.5
#                     lower = upper * -1

#                     # removing all the noise, only want highest peaks and lowest valleys
#                     peaks = data[(data[x] > upper) | (data[x] < lower)]
#                     peaks = peaks.reset_index()

#                     # Making counter to find non-contiguous time blocks, which indicate a new peak or valley
#                     peaks['counter'] = (peaks['index'].shift(-1)-peaks['index'] > 1)
#                     peak_list.append(peaks['counter'].sum())


#                 # Insert dataset results as row in master dataframe
#                 row = [user, age, 'R', file[5], hand[0].capitalize(), data_std[4], data_std[5], data_std[6], data_std[1], data_std[2], data_std[3], peak_list[0], peak_list[1], peak_list[2], peak_list[3], peak_list[4], peak_list[5] ]
#                 master.loc[len(master)] = row

#     # store master df as csv
#     master.to_csv(output_folder + '/result.csv')

