import seaborn as sns
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_roc(labels_,Y_pred_, n_classes, title,  fig_name):
  # Credit: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
  from scipy import interp
  from itertools import cycle

  from sklearn.metrics import roc_curve, auc

  # Plot linewidth.
  lw = 2

  # Compute ROC curve and ROC area for each class
  b = np.zeros((np.array(labels_).size, np.array(labels_).max()+1))
  b[np.arange(np.array(labels_).size),np.array(labels_)] = 1

  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(b[:, i], Y_pred_[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(b.ravel(), Y_pred_.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # Compute macro-average ROC curve and ROC area

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  fig = plt.figure()
  plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
          color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
          label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
          color='navy', linestyle=':', linewidth=4)

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(title)
  plt.legend(loc="lower right")
  fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
  plt.show()
  
import numpy as np

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC,fig_name, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))
    plt.show()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()



def plot_classification_report(classification_report,fig_name, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    # print('plotMat: {0}'.format(plotMat))
    # print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat),fig_name, title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
    
def plot_confusion_matrix(cm,fig_name, classes,
                        normalize=False,
                        title='Confusion matrix',
                        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()
    
def plot_loss(epochs, loss, label, title):
    plt.semilogy(epochs,  loss, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()

def plot_losses(epochs, losses, labels, title, fig_name):
    fig = plt.figure()
    for i in losses:
        plt.semilogy(epochs,  i, label=labels[losses.index(i)])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()

def plot_accuracies(epochs, accuracies, labels, title, fig_name):
    fig = plt.figure()
    for i in accuracies:
        plt.semilogy(epochs,  i, label=labels[accuracies.index(i)])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()
  
def plot_historgram(data, bins, fig_name):
    fig = plt.figure()
    plt.title('Visualize data')
    sns.histplot(data, bins=bins)
    plt.ylabel('Data')
    plt.xlabel('2048-feature vector')
    plt.show() 
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()
    

def plot_sse_silhouette(dataset, clusters, fig_name):
  arr1 = []
  arr2 = []
  for k in clusters: 
    kmean_model = KMeans(n_clusters=k).fit(dataset) 
    preds = kmean_model.predict(dataset) # Calculate prediction from the fitted model
    arr1.append(kmean_model.inertia_) # Get lowset SSE
    arr2.append(silhouette_score(dataset, preds)) # Get silhouette_score using predictions and dataset
  fig = plt.figure()
  plt.subplot(1, 2, 1)
  plt.plot(clusters, arr1, 'bx-') 
  plt.title('The \'elbow method\' in a sum of squared errors') 
  plt.xlabel('K Values') 
  plt.ylabel('Sum of squared error') 
  plt.subplot(1, 2, 2)
  plt.plot(clusters, arr2, 'bx-') 
  plt.title('The silhouette score plot') 
  plt.xlabel('K Values') 
  plt.ylabel('Silhouette score') 
  fig.tight_layout()
  plt.show() 
  fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
  plt.close()
  
def plot_scatter(data, fig_name):
    fig = plt.figure()
    plt.title('Scatter plot')
    plt.scatter(range(len(data)), data)
    plt.ylabel('features')
    plt.xlabel('data')
    plt.show() 
    fig.savefig(fig_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()
    
def plot_feature_vector(data, file_name):
    fig = plt.figure()
    plt.title('2048-feature vector')
    plt.imshow(data)
    plt.show() 
    fig.savefig(file_name, dpi=200, format='png', bbox_inches='tight')
    plt.close()