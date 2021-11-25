import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import kNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensembles import RandomForestClassifier

OUTPUT_TEMPLATE_CLASSFIER = {
    'Bayesian classifier: {bayes:.3g}\n'
    'k-Neighbors classifier: {knn:.3g}\n'
    'Neural Network classifier: {nn:.3g}\n'
    'Random Forest classifier: {forest:.3g}\n'
}

#predicts and saves classification report to report.txt
#each classification is in order from bayes to forest
#all models are in single file for easier viewing
def clf_classification_report(m, X_test):
    y_pred = m.predict(X_test)
    with open('classification_report.txt', 'a') as f:
        f.write((classification_report(y_test, y_pred)))
        f.write('\n\n')

def analyze_data():
    print("Analyzing cleaned data")
    # TODO: read data from files
    data = pd.read_csv('cleaned_data/clean_data.csv')

    # TODO: analyze
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    bayes_clf = GaussianNB()
    #not tested; parameters can change
    knn_clf = kNeighborsClassifier(n_neighbors=4)
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    forest_clf = RandomForestClassifier(n_estimators=100)
    
    models = [bayes_clf, knn_clf, nn_clf, forest_clf]
    
    #fits each model and gets classification report
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        clf_classification_report(m, X_test)

    #prints the score of each model
    #for further analysis look at report.txt
    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes = bayes_clf.score(X_test, y_test),
        knn = knn_clf.score(X_test, y_test),
        nn = nn_clf.score(X_test, y_test),
        forest = forest_clf.score(X_test, y_test),
    ))
