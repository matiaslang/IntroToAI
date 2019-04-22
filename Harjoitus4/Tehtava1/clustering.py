import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import matplotlib.cm as cm
import csv
import copy
from itertools import permutations
import matplotlib.backends.backend_pdf

def Kmeans(X, num_clusters):
    """
    Klusterointi K-means-algoritmilla. Funktio palauttaa klusteroinnin lopputuloksen
    """
    #-------TÄHÄN SINUN KOODI--------
    
    
    #--------------------------------

def Dbscan(X, epsilon, minimum_samples):
    """
    Klusterointi DBSCAN algoritmilla. Funktio palauttaa klusteroinnin lopputuloksen
    """
    #-------TÄHÄN SINUN KOODI--------
    
    
    #--------------------------------

def Agglomerative_clustering(X, num_clusters):
    """
    Klusterointi hierarkkinen klusterointi algoritmilla. Funktio palauttaa klusteroinnin lopputuloksen
    """
    #-------TÄHÄN SINUN KOODI--------
    
    
    #--------------------------------
	
def Load_csv_data(fname):
    """"
    Ladataan data oikeassa muodossa .csv tiedostosta
    """
    data = []
    labels = []
    with open(fname) as csvfile:
        reader = list(csv.reader(csvfile, delimiter=';'))
    for i in range(len(reader)):
        labels.append(int(reader[i][0]))
        data.append(reader[i][1:])
    data = np.asarray(data).astype(float)
    return labels, data
	
def Plot_data(titles, all_datasets,all_labels,num_classes, orig_labels):
    """
    Piirretään datajoukot viiteen pienempään kuvaajaan
    """
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
    st = fig.suptitle(titles[0])
    num_cls = copy.deepcopy(num_classes)
    for i in range(len(num_classes)):
        if num_classes[i] > 1:
            num_cls[i] -= 1
    for i in range(len(all_labels)):
        if num_classes[i] == len(list(set(orig_labels[i]))):
            all_labels[i] = Choose_best_color(all_labels[i], orig_labels[i])  	
    for i in range(len(all_labels[0])):
        ax1.scatter(all_datasets[0][i, 0], all_datasets[0][i, 1], c=[cm.gnuplot(all_labels[0][i]/(num_cls[0]))], s=7, marker='o')
    for i in range(len(all_labels[1])):
        ax2.scatter(all_datasets[1][i, 0], all_datasets[1][i, 1], c=[cm.gnuplot(all_labels[1][i]/(num_cls[1]))], s=7, marker='o')
    for i in range(len(all_labels[2])):
        ax3.scatter(all_datasets[2][i, 0], all_datasets[2][i, 1], c=[cm.gnuplot(all_labels[2][i]/(num_cls[2]))], s=7, marker='o')
    for i in range(len(all_labels[3])):
        ax4.scatter(all_datasets[3][i, 0], all_datasets[3][i, 1], c=[cm.gnuplot(all_labels[3][i]/(num_cls[3]))], s=7, marker='o')
    for i in range(len(all_labels[4])):
        ax5.scatter(all_datasets[4][i, 0], all_datasets[4][i, 1], c=[cm.gnuplot(all_labels[4][i]/(num_cls[4]))], s=7, marker='o')
    ax1.set_title(titles[1])
    ax2.set_title(titles[2])
    ax3.set_title(titles[3])
    ax4.set_title(titles[4])
    ax5.set_title(titles[5])
    ax1.set_xlabel('Accuracy = {}%'.format(accuracy(all_labels[0],orig_labels[0])))
    ax2.set_xlabel('Accuracy = {}%'.format(accuracy(all_labels[1],orig_labels[1])))
    ax3.set_xlabel('Accuracy = {}%'.format(accuracy(all_labels[2],orig_labels[2])))
    ax4.set_xlabel('Accuracy = {}%'.format(accuracy(all_labels[3],orig_labels[3])))
    ax5.set_xlabel('Accuracy = {}%'.format(accuracy(all_labels[4],orig_labels[4])))
    ax6.set_visible(False)
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
	
def Choose_best_color(labels, orig_labels):
    """
    Tämä funktio valitsee klusteroiduille näytteille värin, joka on mahdollisimman lähellä alkuperäistä väriä
    """
    classes = list(set(labels))
    combinations = list(permutations(classes, len(classes)))
    accuracies = []
    labels_array = []
    for i in range(len(combinations)):
        mod_labels = copy.deepcopy(labels)
        for j in range(len(classes)):
            mod_labels = [str(combinations[i][j]) if x==classes[j] else x for x in mod_labels]
        mod_labels = [int(i) for i in mod_labels]
        count = 0
        for k in range(len(mod_labels)):
            if mod_labels[k] == orig_labels[k]:
                count += 1
        accuracies.append(count)
        labels_array.append(mod_labels)
    ind = accuracies.index(max(accuracies))
    return labels_array[ind]
	
def accuracy(pred_labels, orig_labels):
    """
    Tämä funtio laskee klusteroinnin lopputuloksen perusteella luokittelutarkkuuden vertaamalla tulosta alkuperäisiin luokkiin
    """
    classes = list(set(orig_labels))
    samples = 0
    count = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == orig_labels[i]:
            count += 1
        samples += 1
    acc = round(count*100 / samples,3)
    return acc
            
def main():
    colors = []
    pdf = matplotlib.backends.backend_pdf.PdfPages("clustering_result.pdf")
    labels_lsun, data_lsun = Load_csv_data('Lsun.csv')
    labels_engytime, data_engytime = Load_csv_data('EngyTime.csv')
    labels_target, data_target = Load_csv_data('Target.csv')
    labels_twodiamonds, data_twodiamonds = Load_csv_data('TwoDiamonds.csv')
    labels_wingnut, data_wingnut = Load_csv_data('Wingnut.csv')
	#########    Original data    ##############
    print("Load the original data")
    titles = ['Original data','Lsun (classes=3)','EngyTime (classes=2)','Target (classes=6)','TwoDiamonds (classes=2)','Wingnut (classes=2)']
    all_datasets = [data_lsun, data_engytime, data_target, data_twodiamonds, data_wingnut]
    all_labels = [labels_lsun,labels_engytime,labels_target,labels_twodiamonds,labels_wingnut]
    num_classes = [3,2,6,2,2]
    Plot_data(titles, all_datasets, all_labels, num_classes, all_labels)
    pdf.savefig(plt.gcf())
    #########      K-means       ##############
    print("Run K-means algorithm")
    titles_kmeans = ['With Kmeans predicted data','Lsun (classes=3)','EngyTime (classes=2)','Target (classes=6)','TwoDiamonds (classes=2)','Wingnut (classes=2)']
    all_labels_kmeans = [Kmeans(data_lsun, num_classes[0]), Kmeans(data_engytime, num_classes[1]), Kmeans(data_target, num_classes[2]), Kmeans(data_twodiamonds, num_classes[3]), Kmeans(data_wingnut, num_classes[4])]
    Plot_data(titles_kmeans, all_datasets, all_labels_kmeans, num_classes, all_labels)
    pdf.savefig(plt.gcf())
	#########      DBSCAN        ##############
    print("Run DBSCAN algorithm")
    all_labels_DBSCAN = [Dbscan(data_lsun, 0.5, 5), Dbscan(data_engytime, 0.5, 20), Dbscan(data_target, 0.5, 2), Dbscan(data_twodiamonds, 0.25, 70), Dbscan(data_wingnut, 0.5, 70)]
    num_classes_DBSCAN = [len(set(all_labels_DBSCAN[0])),len(set(all_labels_DBSCAN[1])),len(set(all_labels_DBSCAN[2])),len(set(all_labels_DBSCAN[3])),len(set(all_labels_DBSCAN[4]))]
    titles_DBSCAN = ['With DBSCAN predicted data','Lsun (classes={})'.format(num_classes_DBSCAN[0]),'EngyTime (classes={})'.format(num_classes_DBSCAN[1]),'Target (classes={})'.format(num_classes_DBSCAN[2]),'TwoDiamonds (classes={})'.format(num_classes_DBSCAN[3]),'Wingnut (classes={})'.format(num_classes_DBSCAN[4])]
    Plot_data(titles_DBSCAN, all_datasets, all_labels_DBSCAN, num_classes_DBSCAN, all_labels)
    pdf.savefig(plt.gcf())
    ######### Hierarchical clustering #########
    print("Run Hierarchical clustering algorithm")
    titles_agglomerative = ['With Agglomerative Clustering predicted data','Lsun (classes=3)','EngyTime (classes=2)','Target (classes=6)','TwoDiamonds (classes=2)','Wingnut (classes=2)']
    all_labels_agglomerative = [Agglomerative_clustering(data_lsun, num_classes[0]), Agglomerative_clustering(data_engytime, num_classes[1]), Agglomerative_clustering(data_target, num_classes[2]), Agglomerative_clustering(data_twodiamonds, num_classes[3]), Agglomerative_clustering(data_wingnut, num_classes[4])]
    Plot_data(titles_agglomerative, all_datasets, all_labels_agglomerative, num_classes, all_labels)
    pdf.savefig(plt.gcf())
    pdf.close()
    plt.show()	
	
if __name__ == '__main__':
    main()