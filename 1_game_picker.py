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
    print name, aucs, mean_auc
    return name, aucs, mean_auc


def main ():
    run_classifier(clf, X, Y)

if __name__ == '__main__':
    main()
