import pickle
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
from sklearn.externals import joblib
import cv2
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse as ap

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
    alphabet = '123456789'
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

def log_function(values):
    return np.log(np.abs(values))*np.sign(values)

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-C", "--C_parameter", help="Value for C parameter in linear SVM", required="True")
    args = vars(parser.parse_args())
    c_value = float(args["C_parameter"])
    # Ladataan opetusjoukko ja testijoukko
    with open('traindataset.pkl', 'rb') as f:
        traindata = pickle.load(f)
    with open('testdataset.pkl', 'rb') as f:
        testdata = pickle.load(f)
    # Otetaan ylös data ja luokat
    X_train = traindata[0]
    train_labels = traindata[1]
    X_train_features = []
    X_test = testdata[0]
    test_labels = testdata[1]
    X_test_features = []
    for i in range(X_train.shape[0]):
        X_train_features.append(log_function(cv2.HuMoments(cv2.moments(X_train[i])).flatten()))
    for i in range(X_test.shape[0]):
        X_test_features.append(log_function(cv2.HuMoments(cv2.moments(X_test[i])).flatten()))
    clf = svm.SVC(kernel='linear', C=c_value)
    clf.fit(X_train_features, train_labels)
    joblib.dump(clf, "model_linearsvm.pkl", compress=2)
    pdf = matplotlib.backends.backend_pdf.PdfPages("classification_report.pdf")
    accuracy(("Linear-SVM-Classifier",clf),X_test_features,test_labels,pdf)
    pdf.close()
    
    


