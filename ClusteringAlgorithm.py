import numpy as np
import matplotlib.pyplot as plt
import random

dataset = np.genfromtxt(r'iris.csv',delimiter=',')
dataset = np.delete(dataset, 4, axis=1)
dataset = np.delete(dataset, 0, axis=0)

def cluster():
    cluster_center1 = random.randint(0,150)
    cluster_center2 = random.randint(0,150)
    cluster_center3 = random.randint(0,150)
    fuzziness_index_result = np.zeros((8,150))
    for fuzziness_index in range(0,8):
        center_array = np.array([dataset[cluster_center1],dataset[cluster_center2],dataset[cluster_center3]])
        old_center_array = np.zeros((3,4))
        rnk_array = np.zeros((3,150))
        result_array = np.zeros((150))
        fuzziness = (fuzziness_index/2) + 1.5
        convergence = False
        for iteration in range(0,1000):
            point_distance_array = generate_point_distance(dataset, center_array)
            for point in range(0,150):
                for current_cluster in range (0,3):
                    rnk_denom = 0
                    for clusters in range(0,3):
                        if point_distance_array[clusters,point] == 0:
                            new_value = point_distance_array[clusters,point] + 0.00001
                            rnk_denom += ((point_distance_array[current_cluster,point])+0.00001) / (new_value ** (1/(fuzziness - 1)))
                        else:
                            rnk_denom += (((point_distance_array[current_cluster,point])) / ((point_distance_array[clusters,point])))** (1/(fuzziness - 1))
                    rnk_array[current_cluster,point] = np.abs(1/rnk_denom)
            np.copyto(old_center_array, center_array)
            new_center_array = np.zeros_like(center_array)
            for cluster in range(0,3):
                new_center_numerator = np.zeros((4))
                new_center_denom = 0
                for point in range(0,150):
                    for index in range(0,4):
                        new_center_numerator[index] += (rnk_array[cluster,point]**fuzziness) * dataset[point,index]
                    new_center_denom += rnk_array[cluster,point]**fuzziness
                new_center_array[cluster,0] = new_center_numerator[0] / new_center_denom
                new_center_array[cluster,1] = new_center_numerator[1] / new_center_denom
                new_center_array[cluster,2] = new_center_numerator[2] / new_center_denom
                new_center_array[cluster,3] = new_center_numerator[3] / new_center_denom
            np.copyto(center_array, new_center_array)
            if np.array_equal(center_array, old_center_array):
                break
        
        for point in range(0,150):
            membership_array = np.array([rnk_array[0][point],rnk_array[1][point],rnk_array[2][point]])
            result_array[point] = np.argmax(membership_array)+1
            
        fuzziness_index_result[fuzziness_index] = result_array
    return fuzziness_index_result
    

def generate_point_distance(dataset, center_array):
     point_distance_array1 = np.zeros((3,150))
     for point in range(0,150):
         for cluster in range (0,3):
             point_distance_array1[cluster,point] = np.sqrt((dataset[point,0] - center_array[cluster,0])**2 + (dataset[point,1] - center_array[cluster,1])**2+ (dataset[point,2] - center_array[cluster,2])**2+ (dataset[point,3] - center_array[cluster,3])**2)    
     return point_distance_array1
 
def generate_result_matrix(final_result_array):
    results_matrix = np.zeros((8,3))
    for fuzziness_level in range(0,8):
        for cluster_index in range(1,4):
            results_matrix[fuzziness_level, cluster_index-1] = (final_result_array[fuzziness_level] == cluster_index).sum()
    return results_matrix


final_result_array = cluster()
final_matrix = generate_result_matrix(final_result_array)