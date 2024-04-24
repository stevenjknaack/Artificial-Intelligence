""" Steven Knaack; sjknaack; CS540 SU23; P4; clusterer.py """

from parametric_estimator import get_parameter_m_from_input, STATES, INPUT_FILE_NAME
from math import sqrt
from random import randint

K = 6

class Cluster :
    def __init__(self, cluster_nodes) :
        self.cluster_nodes = cluster_nodes
        self.center = None
    
def merge_clusters(cluster1, cluster2) :
    merged_nodes = cluster1.cluster_nodes

    for node in cluster2.cluster_nodes :
        if node in merged_nodes :
            continue
        merged_nodes.append(node)

    merged_cluster = Cluster(merged_nodes)
    return merged_cluster

def _euclidean_distance(vector1, vector2, offset = 2) :
    if len(vector1) != len(vector2) :
        raise(ValueError(f'Things are not right in the distance func: [{vector1},{vector2}]'))
    
    sum = 0
    for i in range(offset, len(vector1)) :
        v1 = vector1[i]
        v2 = vector2[i]

        diff = v1 - v2
        sqrd_diff = diff ** 2

        sum += sqrd_diff

    return sqrt(sum)

def _distance_finder(cluster1, cluster2, optimal_f) :
    nodes1 = cluster1.cluster_nodes
    nodes2 = cluster2.cluster_nodes

    distance = _euclidean_distance(nodes1[0], nodes2[0])
    for i in range(len(nodes1)):
      node1 = nodes1[i]
      for j in range(len(nodes2)) :
          if i == 0 and j == 0 :
              continue
          
          node2 = nodes2[j]
          euc_dist = _euclidean_distance(node1, node2)

          distance = optimal_f(distance, euc_dist)
    
    return distance
  
def complete_linkage(cluster1, cluster2) :
    return _distance_finder(cluster1, cluster2, max)

def single_linkage(cluster1, cluster2) :
    return _distance_finder(cluster1, cluster2, min)

def tie_break(clusters, pair1, pair2) :
    index1, index2, distance1 = pair1
    index3, index4, distance2 = pair2
    cluster1 = clusters[index1]
    cluster2 = clusters[index2]
    cluster3 = clusters[index3]
    cluster4 = clusters[index4]

    cluster1_min = STATES[0]
    for node in cluster1.cluster_nodes :
        state = node[0]
        cluster1_min = min(cluster1_min, state)
    
    cluster2_min = STATES[0]
    for node in cluster2.cluster_nodes :
        state = node[0]
        cluster2_min = min(cluster2_min, state)

    if cluster1_min < cluster2_min :
        pair1_min = cluster1_min
    else :
        pair1_min = cluster2_min

    cluster3_min = STATES[0]
    for node in cluster3.cluster_nodes :
        state = node[0]
        cluster3_min = min(cluster3_min, state)
    
    cluster4_min = STATES[0]
    for node in cluster4.cluster_nodes :
        state = node[0]
        cluster4_min = min(cluster4_min, state)

    if cluster3_min < cluster4_min :
        pair2_min = cluster3_min
    else :
        pair2_min = cluster4_min

    if pair1_min < pair2_min:
        return pair1
    else :
        return pair2


def hierarchical_cluster(data_m, k, distance_f) :
    clusters = []
    for state in data_m :
        data_row = data[state]
        init_cluster = Cluster([data_row])
        clusters.append(init_cluster)

    while len(clusters) > k :
        min_clusters = (-1,-1, 10000)
        for i in range(len(clusters)) :
            cluster1 = clusters[i]

            for j in range(i + 1, len(clusters)) :
                cluster2 = clusters[j]
                distance = distance_f(cluster1, cluster2)

                if abs(distance - min_clusters[2]) <= 0.0001 :
                    new_pair = (i, j, distance)
                    min_clusters = tie_break(clusters, min_clusters, new_pair)
                    print('tie break')
                elif distance < min_clusters[2] :
                    min_clusters = (i, j, distance)
                    
        cluster1 = clusters.pop(min_clusters[1])
        cluster2 = clusters.pop(min_clusters[0])
        
        merged_cluster = merge_clusters(cluster1, cluster2)
        clusters.append(merged_cluster)
    
    return clusters

def cluster_string(clusters, keys) :
    cluster_arr = [-1 for i in range(len(keys))] 
    
    for i in range(len(clusters)) :
        cluster = clusters[i]
        for state in cluster.cluster_nodes :
            state_key = state[0]
            state_index = keys.index(state_key)
            cluster_arr[state_index] = i
    
    return str(cluster_arr)[1:-1]

def k_means_cluster(data_m, k) :
    # generate initial cluster centers and assign points
    clusters = []
    used = [False for i in range(len(data_m))]
    old_centroids = []
    for i in range(k) :
        rand_index = randint(0, len(data_m) - 1)
        while used[rand_index] :
            rand_index = randint(0, len(data_m) - 1)
        used[rand_index] = True

        key = STATES[rand_index]
        rand_cluster_center = data_m[key]
        cluster = Cluster([])
        cluster.center = rand_cluster_center
        clusters.append(cluster)
        old_centroids.append(rand_cluster_center)
    
    reassign_points(clusters, data_m)

    center_change = 10
    num_iterations = 0
    while center_change >= 0.001 and num_iterations < 100 :
        calc_new_centers(clusters, old_centroids)
        reassign_points(clusters, data_m)
        center_change = change_in_centers(old_centroids, clusters)
        num_iterations += 1
    
    return clusters

def change_in_centers(old_centers, clusters) :
    change = 0
    for i in range(len(old_centers)) :
        old_center_i = old_centers[i]
        new_center_i = clusters[i].center
        distance = _euclidean_distance(old_center_i, new_center_i)
        change += distance

    return round(change, 5)

def calc_new_centers(clusters, old_centers) :
    center_len = len(old_centers[0])
    for cluster in clusters :
        nodes = cluster.cluster_nodes
        new_center = [0 for i in range(center_len)]
        for i in range(len(nodes)) :
            node_i = nodes[i]
            for j in range(2, len(new_center)) :
                value_j = node_i[j]
                new_center[j] += value_j / len(nodes)
        old_centers = cluster.center
        cluster.center = new_center

def reassign_points(clusters, data_m) :
    for cluster in clusters:
        cluster.cluster_nodes = []

    for key in data_m :
        node = data_m[key]
        min_cluster = None
        min_center_dis = 1000000

        for cluster in clusters :
            center = cluster.center
            distance = _euclidean_distance(node, center)
            if distance < min_center_dis :
                min_center_dis = distance
                min_cluster = cluster
        
        if min_cluster == None :
            min_cluster = cluster[0]

        min_cluster.cluster_nodes.append(node)
                

def total_distortion(k_means_clusters) :
    sum = 0
    for cluster in k_means_clusters :
        nodes = cluster.cluster_nodes
        center = cluster.center
        for node in nodes :
            distance_sum = 0
            for i in range(2, len(node)) :
                center_i = center[i]
                node_i = node[i]
                distance_sum += (node_i - center_i) ** 2
            sum += distance_sum
    
    return sum


# problem methods and calls

def question5_6(data_m) :
    clusters = hierarchical_cluster(data_m, K, single_linkage)
    single_linkage_str = cluster_string(clusters, STATES)
    print(single_linkage_str)

    clusters_c = hierarchical_cluster(data_m, K, complete_linkage)
    complete_linkage_str = cluster_string(clusters_c, STATES)
    print(complete_linkage_str)

def question7_8_9(data_m) :
    clusters = k_means_cluster(data_m, K)
    clusters_str = cluster_string(clusters, STATES)
    print(clusters_str + '\n')
    
    center_str = ''
    for cluster in clusters :
        center = cluster.center
        for i in range(2, len(center)) :
            component = center[i]
            center_str += str(round(component, 4)) + ','
        center_str = center_str[:-1] + '\n'
    
    print(center_str)

    distortion = total_distortion(clusters)
    print(distortion)

data = get_parameter_m_from_input(INPUT_FILE_NAME)
#question5_6(data)
question7_8_9(data)

