# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import preprocessing
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf

# Opetetaan HOG piirteistä 11 eri luokittelijaa
def ModelRandomGuessing(hog_features, labels, pp):
    model = "RandomGuessing"
    clf = DummyClassifier()
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model0randomguessing.pkl", compress=3)
    return (model, clf)
	
def ModelSVM(hog_features, labels, pp):
    model = "SupportVectorMachine"
    clf = SVC(kernel="rbf")
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model1svm.pkl", compress=3)
    return (model, clf)

def ModelKNN(hog_features, labels, pp):
    model = "k-NearestNeighbors"
    clf = KNeighborsClassifier(n_jobs=-1)
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model2knn.pkl", compress=3)
    return (model, clf)

def ModelDecisionTree(hog_features, labels, pp):
    model = "DecisionTree"
    clf = DecisionTreeClassifier()
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model3decisiontree.pkl", compress=3)
    return (model, clf)

def ModelRandomForest(hog_features, labels, pp):
    model = "RandomForest"
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model4randomforest.pkl", compress=3)
    return (model, clf)

def ModelAdaboost(hog_features, labels, pp):
    model = "AdaptiveBoost"
    clf = AdaBoostClassifier()
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model5adaboost.pkl", compress=3)
    return (model, clf)

def ModelGaussianNB(hog_features, labels, pp):
    model = "GaussianNaiveBayes"
    clf = GaussianNB()
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model6gaussiannb.pkl", compress=3)
    return (model, clf)
    
def ModelSGD(hog_features, labels, pp):
    model = "StochasticGradientDescent"
    clf = SGDClassifier(max_iter=1000,tol=1e-3,n_jobs=-1,verbose=0)
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model7sgd.pkl", compress=3)
    return (model, clf)

def ModelLDA(hog_features, labels, pp):
    model = "LinearDiscriminantAnalysis"
    clf = LinearDiscriminantAnalysis()
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model8lda.pkl", compress=3)
    return (model, clf)

def ModelLogisticRegression(hog_features, labels, pp):
    model = "LogisticRegression"
    clf = LogisticRegressionCV(n_jobs=-1, multi_class='auto', cv=5)
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model9logisticregression.pkl", compress=3)
    return (model, clf)

def ModelMLP(hog_features, labels, pp):
    model = "MultilayerPerceptron"
    clf = MLPClassifier(max_iter=500, verbose=False)
    clf.fit(hog_features, labels)
    joblib.dump((clf, pp), "model10mlp.pkl", compress=3)
    return (model, clf)

def accuracy(modelclf, X_test, Y_test, pdf):
    """
    Tässä funktiossa lasketaan luokittelijoille luokittelutarkkuus, precision, recall, F1-score sekä sekaannusmatriisi
    """
    model, clf = modelclf
    predicted = clf.predict(X_test)
    print("Accuracy for classifier %s: %s\n\n" % (model, accuracy_score(Y_test, predicted)))
    print("Classification report for classifier %s:\n%s\n" % (model, classification_report(Y_test, predicted)))
    print("Confusion matrix for classifier %s:\n%s\n\n" % (model, confusion_matrix(Y_test, predicted)))
    plot_classification_report(classification_report(Y_test, predicted), model, accuracy_score(Y_test, predicted))
    pdf.savefig(plt.gcf())
    plt.close(plt.gcf())
    plotConfusionMatrix(confusion_matrix(Y_test, predicted), model)
    pdf.savefig(plt.gcf())
    plt.close(plt.gcf())
    
def plotConfusionMatrix(confusionmatrix, model):
    """
    Tämä funktio tulostaa sekaannusmatriisin graafisesti
    """
    norm_conf = []
    for i in confusionmatrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, interpolation='nearest')

    width, height = confusionmatrix.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(confusionmatrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '0123456789'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.title("Confusion matrix for {}".format(model))
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

def plot_classification_report(classification_report, model, accuracy):
    """
    Tämä funktio tulostaa luokitteluraportin taulukkoon
    """
    lines = classification_report.split('\n')
    plotMat = []
    class_names = []
    for line in lines[2 : (len(lines) - 5)]:
        t = line.strip().split()
        if len(t) < 2: continue
        v = ['{0:.2f}'.format(float(x)) for x in t[1: len(t) - 1]]
        class_names.append(t[0])
        plotMat.append(v)
    lastline = lines[len(lines)-2]
    t = lastline.strip().split()
    lastline_class_name = ['_'.join(t[0:2])]
    lastline_values = [['{0:.2f}'.format(float(x)) for x in t[2: len(t) - 1]]]
    fig, (ax1, ax2) = plt.subplots(2, sharey=True)
    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['      {}      '.format(class_names[i]) for i in range(len(class_names))]
    the_table = ax1.table(cellText=plotMat, cellLoc='center', rowLabels=yticklabels, colLabels=xticklabels,colWidths=[0.3 for x in range(len(xticklabels))], loc=1)
    the_table1 = ax2.table(cellText=lastline_values, cellLoc='center',rowLabels=lastline_class_name,colWidths=[0.3 for x in range(len(xticklabels))], loc=1)
    ax1.axis('tight')
    ax1.axis('off')
    ax2.axis('tight')
    ax2.axis('off')
    accuracytext = 'Accuracy for classifier {}: {}'.format(model, accuracy)
    plt.text(-0.062,0.00,accuracytext)
    ax1.set(title='Classification report for {}'.format(model))

if __name__=='__main__':
    # Ladataan datajoukko
    X, Y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    X = np.array(X, 'int16')
    Y = np.array(Y, 'int')
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)
    # Lasketaan HOG-piirteet
    list_X_train = []
    for trainsample in X_train:
        fd = hog(trainsample.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2-Hys",visualize=False)
        list_X_train.append(fd)
    X_train = np.array(list_X_train, 'float64')
    pp = preprocessing.StandardScaler().fit(X_train)
    X_train = pp.transform(X_train)
    list_X_val = []
    for valsample in X_val:
        fd = hog(valsample.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2-Hys" ,visualize=False)
        list_X_val.append(fd)
    X_val = np.array(list_X_val, 'float64')
    X_val = pp.transform(X_val)
    # Printataan opetusjoukon ja testijoukon näytteiden lukumäärät
    print ("Count of digits in training dataset", Counter(Y_train))
    print ("Count of digits in test dataset", Counter(Y_val))
    print ("\n\n")
    # Toteutetaan luokittelijat sekä lasketaan niille luokittelutarkkuudet sekä sekaannusmatriisit
    pdf = matplotlib.backends.backend_pdf.PdfPages("classification_report.pdf")
    accuracy(ModelRandomGuessing(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelSVM(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelKNN(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelDecisionTree(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelRandomForest(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelAdaboost(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelGaussianNB(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelSGD(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelLDA(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelLogisticRegression(X_train, Y_train, pp),X_val,Y_val,pdf)
    accuracy(ModelMLP(X_train, Y_train, pp),X_val,Y_val,pdf)
    # Tallennetaan luokittelijoiden luokittelutarkkuudet, luokitteluraportit ja sekaannusmatriisit pdf tiedostoon
    pdf.close()
