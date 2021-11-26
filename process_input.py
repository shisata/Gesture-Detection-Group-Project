import os
import numpy as np
import pandas as pd


input_folder = "input_data"
output_folder = "cleaned_data"


def main():
    # Form master dataframe
    column_names = ['user', 'age', 'dominant_hand','shape', 'hand_used', 'acc_x-std','acc_y-std', 'acc_z-std', 'g-force_x-std', 'g-force_y-std', 'g-force_z-std', 'acc_x-peaks', 'acc_y-peaks', 'acc_z-peaks', 'g-force_x-peaks', 'g-force_y-peaks', 'g-force_z-peaks']
    master = pd.DataFrame(columns=column_names)

    # array of cols in the data files which hold data which will be manipulated
    data_cols=['ax','ay','az','gFx','gFy','gFz']

    # Usings loops to only go through the folder structure
    for user in os.listdir(input_folder):
        if user == 'user1':
            age = 22
        elif user == 'user2':
            age = 56
        elif user == 'user3':
            age = 23
        else:
            age = 19

        for hand in os.listdir(input_folder + "/" + user):
            # Case to ignore git files
            if hand.startswith('.'):
                break
            for file in os.listdir(input_folder + "/" + user + "/" + hand):
                # Case to ignore git files
                if file.startswith('.'):
                    break
                # Getting data from individual csv file and cleaning
                data = pd.read_csv(input_folder + "/" + user + "/" + hand + "/" + file)

                # Array containing standard deviations of columns
                data_std = data.std()
                peak_list = list()

                # Counting peaks
                for x in data_cols:
                    upper = data_std[x] * 1.5
                    lower = upper * -1

                    # removing all the noise, only want highest peaks and lowest valleys
                    peaks = data[(data[x] > upper) | (data[x] < lower)]
                    peaks = peaks.reset_index()

                    # Making counter to find non-contiguous time blocks, which indicate a new peak or valley
                    peaks['counter'] = (peaks['index'].shift(-1)-peaks['index'] > 1)
                    peak_list.append(peaks['counter'].sum())


                # Insert dataset results as row in master dataframe
                row = [user, age, 'R', file[5], hand[0].capitalize(), data_std[4], data_std[5], data_std[6], data_std[1], data_std[2], data_std[3], peak_list[0], peak_list[1], peak_list[2], peak_list[3], peak_list[4], peak_list[5] ]
                master.loc[len(master)] = row

    # store master df as csv
    master.to_csv(output_folder + '/result.csv')

if __name__ == "__main__":
    main()