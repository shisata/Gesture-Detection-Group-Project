import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

input_folder = "cleaned_data"
input_file = "result.csv"
output_folder = "analyzed_data"
report_file = "classification_report.txt"

OUTPUT_TEMPLATE_CLASSIFIER = (
    '-Without Tranformation-\n'
    'Bayesian classifier: {bayes:.3g}\n'
    'k-Neighbors classifier: {knn:.3g}\n'
    'Neural Network classifier: {nn:.3g}\n'
    'Random Forest classifier: {forest:.3g}\n'
)

OUTPUT_TEMPLATE_CLASSIFIER_TRANSFORM = (
    '-With Transformation-\n'
    'Bayesian classifier: {bayes:.3g}\n'
    'k-Neighbors classifier: {knn:.3g}\n'
    'Neural Network classifier: {nn:.3g}\n'
    'Random Forest classifier: {forest:.3g}\n'
)


# Convert DataFrame Columns into numbers so that models can read them
def numerize_data(data):
    # Hand_used: L = 0, R = 1
    if (data['hand_used'] == 'L'):
        data['hand_used'] = 0
    else:
        data['hand_used'] = 1

    # Shape: O = 0, S = 1, V = 2
    if (data['shape'] == 'O'):
        data['shape'] = 0
    elif (data['shape'] == 'S'):
        data['shape'] = 1
    else:
        data['shape'] = 2

    # Dominant_hand: L = 0, R = 1
    if (data['dominant_hand'] == 'L'):
        data['dominant_hand'] = 0
    else:
        data['dominant_hand'] = 1

    # User number only
    start_index = len('user')
    data['user_no'] = int(data['user'][start_index:])
    return data

# Return necessary dataframes for models to analyze (choose only necessary columns for X)
def get_stacked_dataframe(data):
    # X = (data.iloc[:, -12:]) # From acc_x-std to  g-force_z-peaks
    print("Numerizing data for model predictions")
    X = data.iloc[:,-17:] # From user to g-force_z-peaks
 
    X['user_no'] = 0 # Add column to numerize user column
    X = X.apply(numerize_data, axis = 1)
    X = X.drop(columns=['user', 'shape'])
    # print(X)

    y = data['shape']
    return X, y

# Removes existing report file
def remove_report_file(report_file):
    filepath = output_folder + '/' + report_file
    if os.path.exists(filepath):
        print("Removing existing report file: " + report_file)
        os.remove(filepath)


def write_title_to_report(title):
    with open(output_folder + '/' + report_file, 'a') as f:
        f.write('-----' + title + '-----\n')

#predicts and saves classification report to report.txt
#each classification is in order from bayes to forest
#all models are in single file for easier viewing
def clf_classification_report(label, m, X_test, y_test):
    y_pred = m.predict(X_test)
    shape_count = enumerate_y(y_pred)
    plot_predictions(shape_count, label)
    report = classification_report(y_test, y_pred)
    
    with open(output_folder + '/' + report_file, 'a') as f:
        f.write('''\b'''+label+'\n')
        f.write(report)
        f.write('\n\n')

#enumerates through y_test or y_pred to count total instances of 
#each shape
def enumerate_y(y):
    O = 0
    S = 0
    V = 0
    
    for i in (y):
        if(i == 'O'):
            O += 1;
        if(i == 'S'):
            S += 1;
        if(i == 'V'):
            V += 1;
    return [O,S,V]

#taken from stackoverflow
#returns percentage and value
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
        
#plots pie plot for the different shapes
#saves each plot in png file
def plot_predictions(count, label):
    print("Exporting image for prediction: " + label)
    fig = plt.figure(figsize=(5,5))
    shapes = ['O', 'S', 'V']
    plt.pie(count, labels=shapes,textprops={'color':'white', 'weight':'bold', 'fontsize':12.5}, autopct=make_autopct(count))
    plt.title(label)
    
    plt.legend()
    plt.savefig(output_folder + '/' + label)
    # plt.show()
    plt.close(fig)

# produces graphs for data transformations
def plot_scatter_peaks():
    seaborn.set()
    data = pd.read_csv('input_data/user3/right-hand/draw_V_2.csv')
    fsize = (10,5)

    fig1 = plt.figure(figsize=fsize)
    plt.plot(data['gFy'])
    plt.xlabel('time (ms)')
    plt.ylabel('g')
    plt.title('Plotting Raw Data from g-Force y-axis')
    plt.savefig(output_folder + '/' + 'plot_raw_sample_data')
    plt.close(fig1)

    fig2 = plt.figure(figsize=fsize)
    plt.plot(data['gFy'])
    plt.xlabel('time (ms)')
    plt.ylabel('g')
    plt.axhline(y=(data['gFy'].mean()+1.3*data['gFy'].std()), color='r', linestyle='-', label='upper std')
    plt.axhline(y=(data['gFy'].mean()-1.3*data['gFy'].std()), color='r', linestyle='-', label='lower std')
    plt.axhline(y=data['gFy'].mean(), color='g', linestyle=':', label='mean')
    plt.legend()
    plt.title('Plotting Raw Data from g-Force y-axis')
    plt.savefig(output_folder + '/' + 'plot_raw_data_with_std_and_mean')
    plt.close(fig2)

    fig3 = plt.figure(figsize=fsize)
    data = pd.read_csv('cleaned_data/result.csv')
    drawings = ['O', 'S', 'V']
    r=['r','g','b']
    for i in range(3):
        dots = data[data['shape'] == drawings[i]]
        plt.scatter(dots['acc_x-std'], dots['acc_z-std'],color=r[i], label=drawings[i])
    plt.title('Scatter for Standard Deviation in Linear Acceleration')
    plt.xlabel('Standard Deviation of Linear Acceleration of X axis')
    plt.ylabel('Standard Deviation of Linear Acceleration of Z axis')
    plt.legend()
    plt.savefig(output_folder + '/' + 'shape_scatter_in_linear_acc')
    plt.close(fig3)

    fig4 = plt.figure(figsize=fsize)
    data = data[data['acc_x-std'] < 0.2]
    for i in range(3):
        dots = data[data['shape'] == drawings[i]]
        plt.scatter(dots['acc_x-std'], dots['acc_z-std'],color=r[i], label=drawings[i])
    plt.title('Scatter for Standard Deviation in Linear Acceleration')
    plt.xlabel('Standard Deviation of Linear Acceleration of X axis')
    plt.ylabel('Standard Deviation of Linear Acceleration of Z axis')
    plt.legend()
    plt.savefig(output_folder + '/' + 'focused_shape_scatter_in_linear_acc')
    plt.close(fig4)

def analyze_data():
    print("Reading processed data from " + input_folder + ": " + input_file)
    # TODO: read data from files
    data = pd.read_csv(input_folder + '/' + input_file)
    X, y = get_stacked_dataframe(data)
    
    # Making charts for data transformations
    print('Exporting images for Data Transformations')
    plot_scatter_peaks()

    # TODO: analyze
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 1)
    
    test_shape_count = enumerate_y(y_test)
    title = "y_test"
    plot_predictions(test_shape_count, title)
    
    bayes_clf = GaussianNB()
    bayes_clf_pipeline = make_pipeline(
        # PolynomialFeatures(),
        # MinMaxScaler(),
        StandardScaler(),
        # FunctionTransformer(),
        # PCA(10),
        GaussianNB()
    )
    
    knn_clf = KNeighborsClassifier(n_neighbors=7)
    knn_clf_pipeline = make_pipeline(
        # PolynomialFeatures(),
        # MinMaxScaler(),
        StandardScaler(),
        # FunctionTransformer(),
        # PCA(10),
        KNeighborsClassifier(n_neighbors=7)
    )
    
    # nn_clf = MLPClassifier(random_state=1, max_iter=10000)
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(6,), random_state=1, max_iter=10000)
    nn_clf_pipeline = make_pipeline(
        # PolynomialFeatures(),
        # MinMaxScaler(),
        StandardScaler(),
        # FunctionTransformer(),
        # PCA(10),
        MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(6,), random_state=1, max_iter=10000)
    )
    
    forest_clf = RandomForestClassifier(criterion="entropy")
    forest_clf_pipeline = make_pipeline(
        # PolynomialFeatures(),
        # MinMaxScaler(),
        StandardScaler(),
        # FunctionTransformer(),
        # PCA(10),
        RandomForestClassifier(criterion="entropy")
    )
    
    models = [bayes_clf, knn_clf, nn_clf, forest_clf]
    models_pipeline = [bayes_clf_pipeline, knn_clf_pipeline, nn_clf_pipeline, forest_clf_pipeline]
    labels = ["Naive Bayes", "K Neighbor", "Neural Network", "Random Forest"]
    labels_pipeline = ["Naive Bayes With Transformation", "K Neighbor With Transformation", "Neural Network With Transformation", "Random Forest With Transformation"]
    
    #removes existing classification report, preventing new data stacking on old data
    remove_report_file(report_file)

    print("Writing new report file: " + report_file)
    #fits each model and gets classification report
    write_title_to_report("Without Transformation")
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        clf_classification_report(labels[i], m, X_test, y_test)

    #fits each model_pipeline and gets classification report
    write_title_to_report("With Transformation")
    for i, m in enumerate(models_pipeline):
        m.fit(X_train, y_train)
        clf_classification_report(labels_pipeline[i], m, X_test, y_test)

    print("Results of prediction models score")
    #prints the score of each model
    #for further analysis look at report.txt
    #Without Tranformation
    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes = bayes_clf.score(X_test, y_test),
        knn = knn_clf.score(X_test, y_test),
        nn = nn_clf.score(X_test, y_test),
        forest = forest_clf.score(X_test, y_test),
    ))

    #With Tranformation
    print(OUTPUT_TEMPLATE_CLASSIFIER_TRANSFORM.format(
        bayes = bayes_clf_pipeline.score(X_test, y_test),
        knn = knn_clf_pipeline.score(X_test, y_test),
        nn = nn_clf_pipeline.score(X_test, y_test),
        forest = forest_clf_pipeline.score(X_test, y_test),
    ))

    print("Done analysis!! Results can be seen in: " + output_folder)
