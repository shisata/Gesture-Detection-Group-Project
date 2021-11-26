import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

OUTPUT_TEMPLATE_CLASSIFIER = (
    'Bayesian classifier: {bayes:.3g}\n'
    'k-Neighbors classifier: {knn:.3g}\n'
    'Neural Network classifier: {nn:.3g}\n'
    'Random Forest classifier: {forest:.3g}\n'
)

#predicts and saves classification report to report.txt
#each classification is in order from bayes to forest
#all models are in single file for easier viewing
def clf_classification_report(label, m, X_test, y_test):
    y_pred = m.predict(X_test)
    shape_count = enumerate_y(y_pred)
    plot_predictions(shape_count, label)
    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'a') as f:
        f.write('''\b'''+label+'\n')
        f.write(report)
        f.write('\n\n')

#enumerates through y_test or y_pred to count total instances of 
#each shape
def enumerate_y(y):
    O = 0
    S = 0
    V = 0
    
    for i, m in enumerate(y):
        if(m == 'O'):
            O += 1;
        if(m == 'S'):
            S += 1;
        if(m == 'V'):
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
    fig = plt.figure(figsize=(5,5))
    shapes = ['O', 'S', 'V']
    #labels don't seem to show
    plt.pie(count, labels=shapes,textprops={'color':'white', 'weight':'bold', 'fontsize':12.5}, autopct=make_autopct(count))
    plt.title(label)
    
    plt.legend()
    plt.savefig(label)
    plt.show()
    plt.close(fig)
        
def analyze_data():
    print("Analyzing cleaned data")
    # TODO: read data from files
    data = pd.read_csv('cleaned_data/clean_data.csv')
    X = (data.iloc[:, -11:])
    y = data['shape']
    # TODO: analyze
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 1)
    
    test_shape_count = enumerate_y(y_test)
    title = "y_test"
    plot_predictions(test_shape_count, title)
    
#     bayes_clf = make_pipeline(
#         StandardScaler(),
#         GaussianNB()
#     )
    bayes_clf = GaussianNB()
    
#     knn_clf = make_pipeline(
#         StandardScaler(),
#         KNeighborsClassifier(n_neighbors=3)
#     )
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    
#     nn_clf = make_pipeline(
#         StandardScaler(),
#         MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6,), random_state=1)
#     )
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(6,), random_state=1)
    
#     forest_clf = make_pipeline(
#         StandardScaler(),
#         RandomForestClassifier()
#     )
    forest_clf = RandomForestClassifier()
    
    models = [bayes_clf, knn_clf, nn_clf, forest_clf]
    labels = ["Naive Bayes", "K Neighbor", "Neural Network", "Random Forest"]
    
    #fits each model and gets classification report
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        clf_classification_report(labels[i], m, X_test, y_test)

    #prints the score of each model
    #for further analysis look at report.txt
    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes = bayes_clf.score(X_test, y_test),
        knn = knn_clf.score(X_test, y_test),
        nn = nn_clf.score(X_test, y_test),
        forest = forest_clf.score(X_test, y_test),
    ))
