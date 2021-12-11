# Gesture Detection
**NOTE: the current analyzed report files present in this project might be different from the written report, due to prediction model accuracy changes after every run.**
## What is this project about?
This program analyzes the gesture motions that are in the shapes of S, V and O collected from phone. The data will be collected using Physics Toolbox Sensor Suite from many different participants in the form of csv files. From analyzing the data, it will be able to produce different prediction models as well as reports of them.

## How to run this project
 ```
 python3 run_program
 ```
This will run the whole program and display all important stages of processing data.

You can call process_input() in `process_input.py` if you only wish to process the input.

Or can call analyze_data() in `analyze_cleaned_data.py` if you only wish to analyze the cleaned data.

## Folder Structure

 - input_data will be the collected data. You can add more samples as long as they match the following folder structure.

```
user1
    left-hand
        <.csv files>
    right-hand
        <.csv files>
    age.txt
    dominant_hand.txt
user2
    left-hand
        <.csv files>
    right-hand
        <.csv files>
    age.txt
    dominant_hand.txt
user3
    ...
```
 - cleaned_data will be where the program export cleaned input data, which will be called `result.csv`
 - analyzed_data will be where the program export the results of prediction, as well as the charts for each model. The report of models will be called `classification_report.txt`


