import math

import matplotlib.pyplot as plt
import numpy as np


class DataPoint:

    def __init__(self, point, cluster=0):
        self.point = point
        self.cluster = cluster


def point_dist(p1, p2):
    return math.dist((p1.point[0], p1.point[1]), (p2.point[0], p2.point[1]))


def update_assignments(data, centroids):
    for i in range(len(data)):
        point = data[i]
        min_dist = 5000
        for j in range(len(centroids)):
            dist = point_dist(point, DataPoint(centroids[j]))
            if dist < min_dist:
                point.cluster = j
                min_dist = dist




def get_points_in_cluster(data, centroid_id):
    points = []
    for i in range(len(data)):
        if data[i].cluster == centroid_id:
            points.append(data[i])

    return points


def update_centroids(data, old_centroids):
    new_centroids = []

    for i in range(len(old_centroids)):
        points_in_cluter = get_points_in_cluster(data, centroid_id=i)
        num_points = len(points_in_cluter)

        if num_points > 0:
            num_dims = points_in_cluter[0].point.shape[0]
            sum_vector = np.zeros(num_dims)
            for cp in points_in_cluter:
                sum_vector += cp.point

            sum_vector = sum_vector / num_points

            new_centroids.append(sum_vector)
        else:
            new_centroids.append(old_centroids[i])

    return new_centroids


def init_centroids(num_centroids, data):
    rng_default = np.random.default_rng()
    indices_init = rng_default.choice(a=len(data), size=num_centroids, replace=False)

    centroids = []

    for i in range(num_centroids):
        centroids.append(data[indices_init[i]].point)

    return centroids

def load_data():
    blobs_data = np.loadtxt('blobs.dat')

    data_points = []
    for i in range(blobs_data.shape[1]):
        point = blobs_data[:, i]
        data_points.append(DataPoint(point=point))

    return data_points


def refine_centroids(data_points, centroids, num_its):
    curr_centroids = centroids

    centroids_before = [x.cluster for x in data_points]
    for i in range(num_its):
        update_assignments(data_points, curr_centroids)
        centroids_after = [x.cluster for x in data_points]

        if centroids_before == centroids_after:
            #print('Done early', i)
            break


        curr_centroids = update_centroids(data_points, curr_centroids)
        centroids_before = centroids_after

    return curr_centroids


def calculate_intra_cluster_dist(data_points, centroids):

    avicd = 0
    for i in range(len(data_points)):
        p = data_points[i]
        centroid_p = centroids[p.cluster]

        avicd =+ point_dist(p, DataPoint(centroid_p))

    return avicd

def show_clusters(data_points, refined_centroids):
    x = []
    y = []
    centroids = []

    for data in data_points:
        p = data.point
        x.append(p[0])
        y.append(p[1])
        centroids.append(data.cluster)

    fix, axs = plt.subplots(2)
    axs[0].scatter(x, y)
    axs[1].scatter(x, y, c=centroids)
    axs[1].scatter([i[0] for i in refined_centroids], [i[1] for i in refined_centroids])
    plt.show()

def single_test(num_centroids, plot_data=False):
    print(num_centroids)
    data_points = load_data()
    centroids = init_centroids(num_centroids=num_centroids, data=data_points)

    refined_centroids = refine_centroids(data_points, centroids, num_its=5)

    if plot_data:
        show_clusters(data_points, refined_centroids)

    avicd = calculate_intra_cluster_dist(data_points, refined_centroids)

    return avicd


if __name__ == "__main__":
    scree_plot = True

    if not scree_plot:
        single_test(num_centroids=3)
    else:
        avicd = [single_test(i) for i in range(1, 11)]

        plt.plot(avicd)
        plt.show()






