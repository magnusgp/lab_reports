import matplotlib as mpl
# mpl.rcParams['figure.dpi']=100
import numpy as np
from imageio import imread
from skimage.transform import rescale
from skimage.color import rgb2lab, lab2rgb
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import webcolors

# Load image

image_raw = imread('/Users/Magnus/Downloads/Lake2001.jpg')
image_width = 100
image = rescale(image_raw, image_width/image_raw.shape[0], mode='reflect', multichannel=True, anti_aliasing=True)
shape = image.shape
plt.figure()
plt.imshow(image)
plt.axis('off')

X = rgb2lab(image).reshape(-1, 3)
plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=20., azim=120)
ax.set_xlabel("G")
ax.set_ylabel("T")
ax.set_zlabel("A")
image_c = [image.reshape(-1, 3)[i, :] for i in range(image.shape[0] * image.shape[1])]
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], marker='o', c=image_c, linewidths=0)

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
def plot_with_centers(X, y, centers):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=20., azim=120)
    ax.set_xlabel("L")
    ax.set_ylabel("A")
    ax.set_zlabel("B")
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=[colors[i % len(colors)] for i in y_kmeans], 
                 linewidths=0)
    ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], 
                 c=[colors[i % len(colors)] for i in range(K)], marker='+', s=200)

global c
"""
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

requested_colour = (c)
actual_name, closest_name = get_colour_name(requested_colour)"""

def cluster_assignments(X, Y):
    return np.argmin(euclidean_distances(X,Y), axis=1)

K = 3
centers = np.array([X.mean(0) + (np.random.randn(3)/10) for _ in range(K)])
y_kmeans = cluster_assignments(X, centers)
cluster_changes = True
# repeat estimation a number of times (could do something smarter, like comparing if clusters change)
while cluster_changes:
# for i in range(2):
    former_centers = centers.tolist()
    # assign each point to the closest center
    y_kmeans = cluster_assignments(X, centers)

    # move the centers to the mean of their assigned points (if any)
    for i, c in enumerate(centers):
        points = X[y_kmeans == i]
        print(("Cluster: {} , Points: {}").format(i,len(points)))
        if len(points):
            centers[i] = points.mean(0)
    
    if former_centers == centers.tolist():
        break
    # print("Centers: {}".format(centers.tolist()))
    # print("Former_centers {}".format(former_centers))



plot_with_centers(X, y_kmeans, centers)

plt.figure()
plt.imshow(lab2rgb(centers[y_kmeans,:].reshape(shape[0], shape[1], 3)))
plt.axis('off')
plt.show()