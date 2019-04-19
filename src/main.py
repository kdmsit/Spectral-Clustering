# region Packages
import numpy as np
import sys
import os
import math
import networkx as nx
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
from numpy import linalg as LA
# endregion

def cluster(eigenVector,picfilename):
    eigVlen=np.shape(eigenVector)[1]
    positiveEntries = []
    positiveNodes=[]
    negativeEntries = []
    negativeNodes = []
    nodeColorMap = []
    for i in range(0, eigVlen):
        if (eigenVector.item(i) >= 0):
            positiveEntries.append(eigenVector.item(i))
            positiveNodes.append(i+1)
            nodeColorMap.append('blue')
        else:
            negativeEntries.append(eigenVector.item(i))
            negativeNodes.append(i+1)
            nodeColorMap.append('red')
    nx.draw(G, node_color=nodeColorMap, with_labels=True)
    plt.savefig(picfilename);
    plt.close();
    return positiveEntries,positiveNodes,negativeEntries,negativeNodes

if __name__ == '__main__':
    inputfilePath = "/home/kdcse/Documents/Second Semester/ML//Spectral-Clustering/data/karate.edges"
    edges = np.loadtxt(inputfilePath, dtype=int, comments='%')
    G = nx.Graph()
    G.add_edges_from(edges)
    #G = nx.read_gml(inputfilePath)
    Adjacency_lists=G.adj
    node_list = list(G.node)
    print(node_list)
    A = nx.laplacian_matrix(G)
    laplacian_Matrix=A.todense()
    eigenValues, eigenVectors = np.linalg.eigh(laplacian_Matrix)
    EigV=eigenVectors.T
    #eigenValues.sort()
    sortedEigenValueIndex = np.argsort(eigenValues)
    secondSmallestEigenValue = eigenValues[sortedEigenValueIndex[1]]
    secondSmallestEigenVector = EigV[sortedEigenValueIndex[1]]
    kmeans=KMeans(n_clusters=4).fit(np.asanyarray(secondSmallestEigenVector).reshape(-1, 1))
    labels=kmeans.labels_
    print(labels)
    nodeColorMap = []
    for i in range(len(node_list)):
        if (labels[i] == 0):
            nodeColorMap.append('blue')
        elif(labels[i] == 1):
            nodeColorMap.append('red')
        elif(labels[i] == 2):
            nodeColorMap.append('green')
        elif(labels[i] == 3):
            nodeColorMap.append('yellow')
    nx.draw(G, node_color=nodeColorMap, with_labels=True)
    plt.savefig('../output/spectralClustering_karate'+ str(datetime.datetime.now()) + '.png');
    plt.close();
    # outF = open("../output/membership.txt", "w")
    # for line in labels:
    #     outF.write(str(line))
    #     outF.write("\n")
    # outF.close()
    #print(labels)
    #nodeColorMap = ['blue']
    '''for i in range(len(labels)):
        #print(type(labels[i]))
        if(labels[i]==0):
            nodeColorMap.append('blue')
        elif(labels[i]==1):
            nodeColorMap.append('red')
        elif(labels[i]==2):
            nodeColorMap.append('green')
        elif(labels[i]==3):
            nodeColorMap.append('black')
        elif(labels[i]==4):
            nodeColorMap.append('yellow')'''
    # nx.draw(G, node_color=nodeColorMap, with_labels=True)
    # plt.savefig('/home/kdcse/Documents/Second Semester/ML//Spectral-Clustering/output/spectralClustering'+ str(datetime.datetime.now()) + '.png');
    # plt.close();
    '''print("Kmeans :")
    print("\n")
    print(kmeans.labels_)'''
    '''nx.draw(G, node_color=kmeans, with_labels=True)
    plt.savefig(picfilename);
    plt.close();
    print("spectralClustering.png")'''
