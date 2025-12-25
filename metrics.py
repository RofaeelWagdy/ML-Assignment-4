import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment

## Internal Metrics

def calculate_silhouette_score(X, labels):

    """
    Equation: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where:
    - a(i) is the mean cluster distance for point i
    - b(i) is the mean nearest neighbouring cluster distance for point i
    """

    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1 or len(unique_labels) == n_samples:
        return 0.0

    # Pairwise distances (N, N)
    distances = np.linalg.norm(X[:, None] - X[None, :], axis=2)

    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        own_cluster = labels[i]

        if np.sum(labels == own_cluster) == 1:
            silhouette_scores[i] = 0.0
            continue

        # getting distances between data within the same cluster (exclude self)
        same_cluster = (labels == own_cluster)
        same_cluster[i] = False
        a_i = np.mean(distances[i, same_cluster])

        # getting nearest neighbor cluster dstances
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_cluster:
                continue
            other_cluster = (labels == other_label)
            mean_dist = np.mean(distances[i, other_cluster])
            b_i = min(b_i, mean_dist)

        denom = max(a_i, b_i)
        silhouette_scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return np.mean(silhouette_scores), silhouette_scores


def calculate_davies_bouldin(X, labels, centroids):
    """
    Equation:
    DB = (1/n) * sum(max((S_i + S_j) / M_ij))
    where:
    - S_i is the tightness of cluster i (average distance of all data points in cluster rom its centroid).
    - M_ij is the Euclidean distance between the centroids of clusters i and j.
    - n is the number of clusters.
    """
    n_clusters = len(centroids)
    cluster_dispersions = []
    
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            s_i = np.mean(np.linalg.norm(cluster_points - centroids[k], axis=1))
        else:
            s_i = 0
        cluster_dispersions.append(s_i)
        
    db_score = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                # Distance between centroids M_ij
                dist_centroids = np.linalg.norm(centroids[i] - centroids[j])
                if dist_centroids > 0:
                    ratio = (cluster_dispersions[i] + cluster_dispersions[j]) / dist_centroids
                    max_ratio = max(max_ratio, ratio)
        db_score += max_ratio
        
    return db_score / n_clusters


def calculate_calinski_harabasz(X, labels, centroids):
    """
    Equation:
    CH = (BCSS / (k - 1)) / (WCSS / (n - k))
    where:
    - BCSS is the between-cluster sum of squares.
    - WCSS is the within-cluster sum of squares.
    - k is the number of clusters.
    - n is the number of samples.

    BCSS = n_k * (c_k - c_global)^2 for each cluster k
    where: n_k is the number of points in cluster k
           c_k is the centroid of cluster k
           c_global is the global mean of all points

    WCSS = sum((x_i - c_k)^2) for each point x_i in cluster k

    """
    n_samples = X.shape[0]
    n_clusters = len(centroids)
    
    if n_clusters <= 1:
        return 0
    
    # Global mean
    global_mean = np.mean(X, axis=0)
    
    # Between-Cluster sum of Squares 
    ss_b = 0
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        n_k = len(cluster_points)
        ss_b += n_k * np.sum((centroids[k] - global_mean)**2)
        
    # Within-Cluster sum of Squares
    ss_w = 0
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            ss_w += np.sum((cluster_points - centroids[k])**2)
            
    if ss_w == 0:
        return np.inf
        
    score = (ss_b / (n_clusters - 1)) / (ss_w / (n_samples - n_clusters))
    return score


def calculate_wcss(X, labels, centroids):
    wcss = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[k])**2)
    return wcss

#  external metrics 

def calculate_purity(y_true, y_pred):
    """
    Computes clustering purity.
    """
    # Create confusion matrix
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    
    # Map labels to 0..N indices for matrix
    true_map = {label: i for i, label in enumerate(np.unique(y_true))}
    pred_map = {label: i for i, label in enumerate(np.unique(y_pred))}
    
    for t, p in zip(y_true, y_pred):
        contingency_matrix[true_map[t], pred_map[p]] += 1
        
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def combs(n, k): # Compute n choose k nck
        if k < 0 or k > n: return 0
        if k == 0 or k == n: return 1
        if k > n // 2: k = n - k
        
        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res

def calculate_ari(y_true, y_pred):
    """
    Equation:
    ARI = (Index - Expected Index) / (Max Index - Expected Index)
    """

    # Contingency table
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    table = np.zeros((len(classes), len(clusters)))
    
    c_map = {val: i for i, val in enumerate(classes)}
    k_map = {val: i for i, val in enumerate(clusters)}
    
    for t, p in zip(y_true, y_pred):
        table[c_map[t], k_map[p]] += 1
        
    # Sum over rows (ni.) and columns (n.j)
    nis = np.sum(table, axis=1)
    njs = np.sum(table, axis=0)
    n = len(y_true)
    
    # Index Calculation
    sum_nij_comb = sum([combs(nij, 2) for row in table for nij in row])
    sum_ni_comb = sum([combs(ni, 2) for ni in nis])
    sum_nj_comb = sum([combs(nj, 2) for nj in njs])
    total_comb = combs(n, 2)
    
    expected_index = (sum_ni_comb * sum_nj_comb) / total_comb
    max_index = (sum_ni_comb + sum_nj_comb) / 2
    
    if max_index == expected_index:
        return 0
        
    ari = (sum_nij_comb - expected_index) / (max_index - expected_index)
    return ari

def calculate_nmi(y_true, y_pred):
    """
    Normalized Mutual Information.
    """
    n = len(y_true)
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    
    # Probabilities
    p_classes = np.array([np.sum(y_true == c) / n for c in classes])
    p_clusters = np.array([np.sum(y_pred == k) / n for k in clusters])
    
    # Entropy H(Y) and H(C)
    h_y = -np.sum(p_classes * np.log(p_classes + 1e-10))
    h_c = -np.sum(p_clusters * np.log(p_clusters + 1e-10))
    
    # Mutual Information I(Y;C)
    mi = 0
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            # Intersection probability
            intersect = np.sum((y_true == c) & (y_pred == k)) / n
            if intersect > 0:
                mi += intersect * np.log(intersect / (p_classes[i] * p_clusters[j]) + 1e-10)
                
    return 2 * mi / (h_y + h_c)
