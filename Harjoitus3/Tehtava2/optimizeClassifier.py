from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

def log_function(values):
    return np.log(np.abs(values))*np.sign(values)
	
if __name__=='__main__':
    # Ladataan opetusjoukko
    with open('traindataset.pkl', 'rb') as f:
        traindata = pickle.load(f)
    # Otetaan ylös data ja luokat
    X_train = traindata[0]
    labels = traindata[1]
    X_train_features = []
    for i in range(X_train.shape[0]):
        X_train_features.append(log_function(cv2.HuMoments(cv2.moments(X_train[i])).flatten()))

    clf = svm.SVC(kernel='linear')
    Cs = np.logspace(-10, 0, 11)

    scores = []
    scores_min = []
    scores_max = []
    for C in Cs:
        clf.C = C
        cross_val_value = cross_val_score(clf, X_train_features, labels, cv=5, n_jobs=1)
        scores.append(np.mean(cross_val_value))
        scores_max.append(np.max(cross_val_value))
        scores_min.append(np.min(cross_val_value))

    # Plotataan kuvaaja
    plt.figure(1, figsize=(12, 8))
    plt.clf()
    plt.semilogx(Cs, scores)
    plt.semilogx(Cs, np.array(scores_max), 'b--')
    plt.semilogx(Cs, np.array(scores_min), 'r--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('Accuracy')
    plt.xlabel('Penalty parameter C')
    plt.title('Cross-validation for penalty parameter C (cv=5)')
    plt.ylim(0, 1.0)
    plt.legend(['Average Accuracy', 'Maximum Accuracy', 'Minimum accuracy'], loc='upper left')
    plt.savefig("crossvalidation.png")
    plt.show()
    


