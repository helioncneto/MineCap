# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 16:31:54 2018

@author: Helio

Modulos para criar gráficos.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from matplotlib.backends.backend_pdf import PdfPages


def carregar_dataset(dt='dataset_fluxo_bc.csv'):
    #Carregar Dataset
    fluxos = pd.read_csv(dt,usecols=(4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44))
    fluxos_classes = pd.read_csv(dt,usecols=[45])

    #Pre-processamento do dataset

    sc = MinMaxScaler(feature_range = (0, 1))
    fluxos_scaled = sc.fit_transform(fluxos)


    #Convertendo  os datasets em variáveis do numpy
    x = np.array(fluxos_scaled)
    y = np.array(fluxos_classes)
    
    return x, y

def pca(x):
    pca = PCA(n_components=35)
    x_pca = pca.fit_transform(x)
    return x_pca


def rna(x, y):
    mper = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(21,), random_state=1, 
                    learning_rate='constant', learning_rate_init=0.01, max_iter=500, 
                    activation='relu', momentum=0.8, verbose=False, tol=0.000001)

    #Treinamento com o teste retirado da base de treinamento
    mper.fit(x, y)
    return mper

def d_tree(x, y, md=3):
    dt = DecisionTreeClassifier(random_state=1986, criterion='entropy', max_depth=md)
    #Treinamento com o teste retirado da base de treinamento
    dt.fit(x, y)
    return dt

def k_neig(x, y, k=5):
    kneig = KNeighborsClassifier(n_neighbors=k)
    #Treinamento com o teste retirado da base de treinamento
    kneig.fit(x, y)
    return kneig

def proba(mod, x, x_test, y):
    mod_isotonic = CalibratedClassifierCV(mod, cv=2, method='isotonic')
    mod_isotonic.fit(x, y)
    prob_pos_isotonic = mod_isotonic.predict_proba(x_test)[:]
    return prob_pos_isotonic

def cross_val(mod, x, y, cv=20, score='accuracy'):
    scores = cross_val_score(mod, x, y, cv=cv,scoring=score)
    return scores

def matriz_con(y, saida):
    cm = confusion_matrix(y, saida)
    return cm

def resultado(mod, x, y, saida):
    tn, fp, fn, tp = confusion_matrix(y, saida).ravel()
    acuracia = mod.score(x, y)
    precision = tp / (tp + fp)
    sensibilidade = tp / (tp + fn)
    especificidade = tn / (tn + fp)
    return tn, fp, fn, tp, acuracia, precision, sensibilidade, especificidade

def plot_roc(y, prob, titulo=''):
    
    a = []
    for i in range(len(y)):
        if y[i] == 0:
            a.append([1, 0])
        else:
            a.append([0, 1])
    a = np.array(a)


    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(a[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(a.ravel(), prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='Curva ROC (AUC = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title(titulo)
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_multi(y1, y2, y3, prob1, prob2, prob3, alg1, alg2, alg3, titulo=''):
    
    a1 = []
    for i in range(len(y1)):
        if y1[i] == 0.0:
            a1.append([1, 0])
        else:
            a1.append([0, 1])
    a1 = np.array(a1)

    a2 = []
    for i in range(len(y2)):
        if y2[i] == 0.0:
            a2.append([1, 0])
        else:
            a2.append([0, 1])
    a2 = np.array(a2)

    a3 = []
    for i in range(len(y3)):
        if y3[i] == 0.0:
            a3.append([1, 0])
        else:
            a3.append([0, 1])
    a3 = np.array(a3)

    fpr1, fpr2, fpr3 = dict(), dict(), dict()
    tpr1, tpr2, tpr3 = dict(), dict(), dict()
    roc_auc1, roc_auc2, roc_auc3 = dict(), dict(), dict()

    for i in range(1):
        fpr1[i], tpr1[i], _ = roc_curve(a1[:, i], prob1[:, i])
        roc_auc1[i] = auc(fpr1[i], tpr1[i])

    # Compute micro-average ROC curve and ROC area
    fpr1["micro"], tpr1["micro"], _ = roc_curve(a1.ravel(), prob1.ravel())
    roc_auc1["micro"] = auc(fpr1["micro"], tpr1["micro"])
    
    for i in range(1):
        fpr2[i], tpr2[i], _ = roc_curve(a2[:, i], prob2[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])

    # Compute micro-average ROC curve and ROC area
    fpr2["micro"], tpr2["micro"], _ = roc_curve(a2.ravel(), prob2.ravel())
    roc_auc2["micro"] = auc(fpr2["micro"], tpr2["micro"])

    for i in range(1):
        fpr3[i], tpr3[i], _ = roc_curve(a3[:, i], prob3[:, i])
        roc_auc3[i] = auc(fpr3[i], tpr3[i])

    # Compute micro-average ROC curve and ROC area
    fpr3["micro"], tpr3["micro"], _ = roc_curve(a3.ravel(), prob3.ravel())
    roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])
    
#    for i in range(1):
#        fpr3[i], tpr3[i], _ = roc_curve(a[:, i], prob3[:, i])
#        roc_auc3[i] = auc(fpr3[i], tpr3[i])

    # Compute micro-average ROC curve and ROC area
#    fpr3["micro"], tpr3["micro"], _ = roc_curve(a.ravel(), prob3.ravel())
#    roc_auc3["micro"] = auc(fpr3["micro"], tpr3["micro"])
    
    
    plt.figure()
    lw = 2
    fig, ax = plt.subplots()
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.plot(fpr1[0], tpr1[0], color='darkorange', lw=lw, label=alg1+' (AUC = %0.2f)' % roc_auc1[0])
    plt.plot(fpr2[0], tpr2[0], color='green', lw=lw, label=alg2+' (AUC = %0.2f)' % roc_auc2[0])
    plt.plot(fpr3[0], tpr3[0], color='red', lw=lw, label=alg3+' (AUC = %0.2f)' % roc_auc3[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.0, 1.0])
    plt.ylim([0.0, 1.05])
  
    '''
    ax.annotate('KNN', xy=(0.25,0.4), xytext=(0.15,0.5),
                arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
                )
    
    ax.annotate('CDT', xy=(0.15, 0.81), xytext=(0.1,0.67), 
                arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
            )
    ax.annotate('MLP', xy=(0.35, 0.96), xytext=(0.35,0.85), 
                arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
            )
    
    ax.set_ylim(0,1)
    '''
    '''
    ax.annotate('KNN', xy=(0.15,1), xytext=(0.1,0.85),
                arrowprops=dict(arrowstyle="->",connectionstyle="arc3")
                )
    
    ax.annotate('CDT', xy=(0.15, 0.81), xytext=(0.1,0.67), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    ax.annotate('MLP', xy=(0.35, 0.96), xytext=(0.39,0.89), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    
    ax.set_ylim(0,1)
     '''
    
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title(titulo)
    plt.legend(loc="lower right")
    #plt.savefig('plot.pdf', bbox_inches='tight')
    plt.show()

    
'''
    plotfile = 'plot.pdf'               ## nome do arquivo que vc quer salvar o gráfico
    pp = PdfPages(plotfile)       ## a partir daqui é o conjunto de comandos que fazem o gráfico ser salvo no arquivo PDF
    pp.savefig()
    pp.close()
    plt.clf()
    plt.cla()
    plt.close()
'''

def plot_pca_carac(n_mlp, n_knn, n_cdp, p_mlp, p_knn, p_cdp, score='',titulo=''):
    '''
    y_axis = y
    x_axis = x
    plt.title(titulo)
    width_n = 0.2
    bar_color = 'yellow'
    plt.bar(x_axis, y_axis, width=width_n, color=bar_color)
    plt.show()
    '''

    grupos = 3
    normal = (n_mlp, n_knn, n_cdp)
    pca = (p_mlp, p_knn, p_cdp)
    
    fig, ax = plt.subplots()
    indice = np.arange(grupos)
    bar_larg = 0.2
    transp = 0.7
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.bar(indice, normal, bar_larg, alpha=transp, color='red', label='Normal', edgecolor='black', linewidth=1)
    plt.bar(indice + bar_larg, pca, bar_larg, alpha=transp, color='lightblue', label='PCA', edgecolor='black', linewidth=1)

    #plt.xlabel('Modelos de Aprendizado de Máquina') 
    plt.ylabel(score) 
    plt.title(titulo) 
    plt.xticks(indice + bar_larg, ('RNA', 'KNN', 'DTC')) 
    axes = plt.gca()
    axes.set_ylim([0,1])

    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plotfile = 'plot.pdf'               ## nome do arquivo que vc quer salvar o gráfico
    pp = PdfPages(plotfile)       ## a partir daqui é o conjunto de comandos que fazem o gráfico ser salvo no arquivo PDF
    pp.savefig()
    pp.close()
    plt.clf()
    plt.cla()
    plt.close()
    
    #plt.show()
    
def f_importance(x, y):
    # Create decision tree classifer object
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)

    # Train model
    model = clf.fit(x, y)

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = np.array(['proto','total_fpackets','total_fvolume','total_bpackets','total_bvolume','min_fpktl','mean_fpktl','max_fpktl','std_fpktl','min_bpktl','mean_bpktl','max_bpktl','std_bpktl','min_fiat','mean_fiat','max_fiat','std_fiat','min_biat','mean_biat','max_biat','std_biat','duration','min_active','mean_active','max_active','std_active','min_idle','mean_idle','max_idle','std_idle','sflow_fpackets','sflow_fbytes','sflow_bpackets','sflow_bbytes','fpsh_cnt','bpsh_cnt','furg_cnt','burg_cnt','total_fhlen','total_bhlen','dscp','classe'])

    # Create plot
    plt.figure()
    plt.rcParams['figure.figsize'] = (10,8)

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(x.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x.shape[1]), names, rotation=90)

    # Show plot
    plt.show()