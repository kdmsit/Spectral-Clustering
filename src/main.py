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
    inputfilePath = "/home/kdcse/Documents/Second Semester/ML//Spectral-Clustering/data/football.gml"
    G = nx.read_gml(inputfilePath)
    Adjacency_lists=G.adj
    node_list = list(G.node)
    A = nx.laplacian_matrix(G)
    laplacian_Matrix=A.todense()
    eigenValues, eigenVectors = np.linalg.eig(laplacian_Matrix)
    EigV=eigenVectors.T
    #eigenValues.sort()
    sortedEigenValueIndex = np.argsort(eigenValues)
    secondSmallestEigenValue = eigenValues[sortedEigenValueIndex[1]]
    secondSmallestEigenVector = EigV[sortedEigenValueIndex[1]]
    positiveEntries, positiveNodes, negativeEntries, negativeNodes = cluster(secondSmallestEigenVector[0],'/home/kdcse/Documents/Second Semester/ML//Spectral-Clustering/output/spectralClustering'+ str(datetime.datetime.now()) + '.png')
    print("spectralClustering.png")