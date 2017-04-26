import scipy
import numpy
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def run_classifier(clf, X, Y):
    skf = StratifiedKFold(n_splits=5)
    aucs = []

    for train, test in skf.split(X, Y):
        clf.fit(X[train], Y[train])
        prediction = clf.predict_proba(X[test])
        aucs.append(roc_auc_score(Y[test], prediction[:, 1]))
    name, aucs, mean_auc = clf.__class__.__name__, aucs, numpy.mean(aucs)
    #print Area-Under-the-Curve information
    print name, aucs, mean_auc
    return name, aucs, mean_auc

def extract_labels(Arr):
    #TODO take the labels out of the downloaded CSV, return an array of labels

def extract_metrics (Arr):
    #TODO extract data that we think will be sufficiently predictive
    # Return 2D array of statistics
    # Data rows should correspond to games and labels, columns should be useful metrics

def load_data:
    #TODO Extract data from CSV and put in 2D array

def main ():
    #TODO Pick a type of classifier, run classifier on extracted data   
    run_classifier(clf, X, Y)

if __name__ == '__main__':
    main()
