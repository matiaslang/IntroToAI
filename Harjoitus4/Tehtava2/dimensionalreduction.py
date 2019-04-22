from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding, Isomap
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from skimage.transform import resize
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import matplotlib.cm as cm
import matplotlib
import math
import matplotlib.backends.backend_pdf
	
def Load_wines_dataset():
    """
    Ladataan datajoukko wines tiedostosta wines.pkl
    """
    with open("wines.pkl", "rb") as f:
        data = pickle.load(f)
    X,labels = data[0], data[1]
    real_classes = ['wine1', 'wine2', 'wine3']
    return X, labels, real_classes

def Load_Fashion_MNIST_dataset():
    """
    Ladataan datajoukko fashionMNIST tiedostosta fashionmnist.pkl
    """
    with open("fashionmnist.pkl", "rb") as f:
        data = pickle.load(f)
    images,labels = data[0], data[1]
    images = np.asarray(images)
    real_classes = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return images, labels, real_classes

def Load_COIL20_dataset():
    """
    Ladataan datajoukko COIL20 tiedostosta coil20.pkl
    """
    with open("coil20.pkl", "rb") as f:
        data = pickle.load(f)
    images,labels = data[0], data[1]
    images = np.reshape(images, (1440, 1024))
    real_classes = ['rubber duck','blocks1','toycar1','figure','medicine1','toycar2','blocks2','baby powder','medicine2','vaseline','blocks3','cup1','piggy bank','cylinder','can','jar','pot','cup2','toycar3','plastic jar']
    return images, labels, real_classes
	
def MinMax_normalization(data):
    """
    Toteutetaan datan normalisointi niin, että arvot sijoittuvat x-akselin ja y-akselin välille [0,1]
    """
    #-------TÄHÄN SINUN KOODI--------
    X_min, X_max = np.min(data, 0), np.max(data, 0)
    X_scaled = (data - X_min) / (X_max - X_min)
    return X_scaled
    #--------------------------------
	
def Train_model(X_data,  params, model):
    """
    Käytetään valittua dimensionaalisuuden vähentämismenetelmää korkeadimensioiseen dataan ja palautetaan normalisoidut kaksiulotteiset arvot sekä laskenta-aika
    """
    t0 = time()
    if model == 'PCA':
        X_reduced_data = PCA(n_components=2).fit_transform(X_data)
    elif model == 'MDS':
        X_reduced_data = MDS(n_jobs=-1).fit_transform(X_data)
    elif model == 'LLE':
        X_reduced_data = LocallyLinearEmbedding(method='modified',n_neighbors=params[0], n_jobs=-1).fit_transform(X_data)
    elif model == 'ISOMAP':
        X_reduced_data = Isomap().fit_transform(X_data)
    elif model == 'TSNE':
        X_reduced_data = TSNE(n_components=2, metric='sqeuclidean').fit_transform(X_data)
    elif model == 'UMAP':
        X_reduced_data = umap.UMAP(n_neighbors=params[1], min_dist=params[2], metric='correlation').fit_transform(X_data)
    #-------TÄHÄN SINUN KOODI--------
    t1 = time()
    deltatime = t1 - t0
    X_scaled = MinMax_normalization(X_reduced_data)
    return X_scaled, deltatime
    #--------------------------------
	
def Evaluation(X_data, labels):
    """
    Luodaan yksinkertainen knn-luokittelija mittaamaan ja arvioimaan dimensionaasuuden menetelmällä vähennetyn datan rakenteen kompleksisuutta
    """
    clf = KNeighborsClassifier(n_neighbors=49).fit(X_data, labels)
    y_pred = clf.predict(X_data)	
    acc = accuracy_score(labels, y_pred)
    return round(acc*100,3)

def Draw_example_pictures_to_figure(X_imgs, X_data, ax):
    """
    Piirretään esimerkki näytteitä kuvaikkunoissa kuvaajaan valitun minimietäisyyden välein havainnollistamaan datan sisältöä
    """
    cols = int(math.sqrt(X_imgs.shape[1]))
    X_images = np.reshape(X_imgs, (-1, cols, cols))
    shown_images = np.array([[1., 1.]])
    for i in range(X_data.shape[0]):
        dist = np.sum((X_data[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:     # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X_data[i]]]
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(resize(X_images[i], (30,30), order=1, preserve_range=True, anti_aliasing=True, mode='constant'), cmap=plt.cm.gray),X_data[i])
        ax.add_artist(imagebox)
	
def Draw_datasamples_to_figure(X_scaled, labels, axis):
    """
    Plotataan datanäytteet kuvaajaan. Datanäytteiden luokkia on kuvattu numeroin ja värein
    """
    markers = ['${}$'.format(i) for i in labels]
    num_cls = len(list(set(labels)))
    for (X_plot, Y_plot, y, label) in zip(X_scaled[:,0], X_scaled[:,1], markers, labels):
        axis.scatter(X_plot, Y_plot, color=cm.gnuplot(int(label)/num_cls), marker=y, s=60)
		
def Draw_datasamples_and_pictures(X_data, X_scaled, labels, real_classes, method, title, number, pdf, time):
    """
    Plotataan datanäytteet sekä kuvaikkunat samaan kuvaajaan
    """
    fig, ax = plt.subplots(figsize=(15, 10), dpi=40)
    Draw_example_pictures_to_figure(X_data, X_scaled, ax)
    Draw_datasamples_to_figure(X_scaled, labels, ax)
    plt.title('{} dataset with images for {} ({}/6)'.format(title, method, number), fontsize=16)
    plt.subplots_adjust(left=0.04, bottom=0.11, right=0.89, top=0.92, wspace=0.11, hspace=0.26)
    num_cls = len(list(set(labels)))
    markers = ['${}$'.format(i) for i in range(1, num_cls+1)]
    patches = [plt.plot([],[], marker=markers[i], color=cm.gnuplot((int(i)+1)/num_cls), linestyle = 'None', label=real_classes[i])[0]  for i in range(len(real_classes))]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_scaled, labels),2),round(time,2)))
    pdf.savefig(plt.gcf())
    plt.close(plt.gcf())
		
def Draw_all_figures(X_data, labels, real_classes, title, params, pdf, print_pictures=True):
    """
    Suoritetaan valitulle datajoukolle kaikilla vähentämismenetelmillä kuvaajien piirtäminen
    """
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(15,10), dpi=40)
    X_data_pca, time1 = Train_model(X_data, params, 'PCA')
    Draw_datasamples_to_figure(X_data_pca, labels, ax1)
    ax1.set_title('PCA', fontsize=16)
    ax1.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_data_pca, labels),2),round(time1,4)))
    print("1/6")
    X_data_mds, time2 = Train_model(X_data, params, 'MDS')
    Draw_datasamples_to_figure(X_data_mds, labels, ax2)
    ax2.set_title('MDS', fontsize=16)
    ax2.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_data_mds, labels),2),round(time2,2)))
    print("2/6")   
    X_data_lle, time3 = Train_model(X_data, params, 'LLE')
    Draw_datasamples_to_figure(X_data_lle, labels, ax3)
    ax3.set_title('LLE', fontsize=16)
    ax3.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_data_lle, labels),2),round(time3,2)))	
    print("3/6")    
    X_data_isomap, time4 = Train_model(X_data, params, 'ISOMAP')
    Draw_datasamples_to_figure(X_data_isomap, labels, ax4)
    ax4.set_title('ISOMAP', fontsize=16)
    ax4.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_data_isomap, labels),2),round(time4,4)))
    print("4/6")    
    X_data_tsne, time5 = Train_model(X_data, params, 'TSNE')
    Draw_datasamples_to_figure(X_data_tsne, labels, ax5)
    ax5.set_title('T-SNE', fontsize=16)
    ax5.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_data_tsne, labels),2),round(time5,2)))	
    print("5/6")    
    X_data_umap, time6 = Train_model(X_data, params, 'UMAP')
    Draw_datasamples_to_figure(X_data_umap, labels, ax6)
    ax6.set_title('UMAP', fontsize=16)
    ax6.set_xlabel('Evaluated accuracy = {}%\ntime:{} sec'.format(round(Evaluation(X_data_umap, labels),2),round(time6,2)))
    fig.suptitle("2D-visualization of {} dataset".format(title), fontsize=20)
    plt.subplots_adjust(left=0.04, bottom=0.11, right=0.89, top=0.92, wspace=0.11, hspace=0.26)
    print("6/6")
	
    num_cls = len(list(set(labels)))
    markers = ['${}$'.format(i) for i in range(1, num_cls+1)]
    patches = [plt.plot([],[], marker=markers[i], color=cm.gnuplot((int(i)+1)/num_cls), linestyle = 'None', label=real_classes[i])[0]  for i in range(len(real_classes))]
    ax3.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    pdf.savefig(plt.gcf())
    plt.close(plt.gcf())
	
    if print_pictures:
        Draw_datasamples_and_pictures(X_data, X_data_pca, labels, real_classes, 'PCA', title, '1', pdf, time1)
        Draw_datasamples_and_pictures(X_data, X_data_mds, labels, real_classes, 'MDS', title, '2', pdf, time2)
        Draw_datasamples_and_pictures(X_data, X_data_lle, labels, real_classes, 'LLE', title, '3', pdf, time3)
        Draw_datasamples_and_pictures(X_data, X_data_isomap, labels, real_classes, 'ISOMAP', title, '4', pdf, time4)
        Draw_datasamples_and_pictures(X_data, X_data_tsne, labels, real_classes, 'T-SNE', title, '5', pdf, time5)
        Draw_datasamples_and_pictures(X_data, X_data_umap, labels, real_classes, 'UMAP', title, '6', pdf, time6)	
    
def main():
    X_data_wines, labels_wines, real_labels_wines = Load_wines_dataset()
    X_data_fashionMNIST, labels_fashionMNIST, real_labels_fashionMNIST = Load_Fashion_MNIST_dataset()
    X_data_COIL20, labels_COIL20, real_labels_COIL20 = Load_COIL20_dataset()

    pdf1 = matplotlib.backends.backend_pdf.PdfPages("reduction_wines.pdf")
    print('wines')
    Draw_all_figures(X_data_wines, labels_wines, real_labels_wines, "wines", [15,23,0.2], pdf1, False)
    pdf1.close()
    pdf2 = matplotlib.backends.backend_pdf.PdfPages("reduction_coil20.pdf")
    print('COIL20')
    Draw_all_figures(X_data_COIL20, labels_COIL20, real_labels_COIL20, "COIL20", [68,4,0.0003], pdf2, True)
    pdf2.close()
    pdf3 = matplotlib.backends.backend_pdf.PdfPages("reduction_fashionmnist.pdf")
    print('fashionMNIST')
    Draw_all_figures(X_data_fashionMNIST, labels_fashionMNIST, real_labels_fashionMNIST, "fashion MNIST", [14,11,0.17], pdf3, True)
    pdf3.close()

if __name__ == '__main__':
    main()