import cv2
import numpy as np
from skimage import io
from sklearn.cluster import KMeans

#open the two images

im1 = io.imread('output/crop.mp4/08-01-21/car2_2.jpg')[:, :, :-1]
im2 = io.imread('output/crop.mp4/08-01-21/car2_4.jpg')[:, :, :-1]

#compare avg color

def compare_avg_colors(im1, im2):
	tolerance = 20

	avg1 = im1.mean(axis=0).mean(axis=0)
	avg2 = im2.mean(axis=0).mean(axis=0)

	rangeLo = [avg1[0] - tolerance, avg1[1] - tolerance]
	rangeUp = [avg1[0] + tolerance, avg1[1] + tolerance]

	if ((avg2[0] > rangeLo[0] and avg2[0] < rangeUp[0]) and (avg2[1] > rangeLo[1] and avg2[1] < rangeUp[1])):
		print('The average color of the two images match')

		return True
	else:
		print('The average color of the two images does not match')

		return False

#compare_avg_colors(im1, im2)

#compare primary color(s)

def compare_primary_colors(im1, im2):
	tolerance = 10

	pixels1 = np.float32(im1.reshape(-1, 3))
	pixels2 = np.float32(im2.reshape(-1, 3))

	n_colors = 5
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
	flags1 = cv2.KMEANS_RANDOM_CENTERS

	_, labels, palette1 = cv2.kmeans(pixels1, n_colors, None, criteria, 10, flags1)
	_, counts = np.unique(labels, return_counts=True)

	flags2 = cv2.KMEANS_RANDOM_CENTERS

	_, labels, palette2 = cv2.kmeans(pixels2, n_colors, None, criteria, 10, flags2)
	_, counts = np.unique(labels, return_counts=True)

	dominant1 = palette1[np.argmax(counts)]
	dominant2 = palette2[np.argmax(counts)]

	similar = True
	i = 0

	for val in dominant1:
		if (not(dominant2[i] > val - tolerance and dominant2[i] < val + tolerance)):
			similar = False
			break

		++i

	if similar:
		print('The dominant colors are similar')

		return True
	else:
		print('The dominant colors are not similar')

		return False

#compare_primary_colors(im1, im2)

#do both at once

if (compare_avg_colors(im1, im2) and compare_primary_colors(im1, im2)):
	print('The colors seems to show the similar car based on the color values')
else:
	print('The colors do not seem to show the same car based on the color values')

#visualization of dominant colors

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype = np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0

    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end

    return rect

# Load image and convert to a list of pixels
image1 = cv2.imread('output/crop.mp4/08-01-21/car2_2.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
reshape1 = image1.reshape((image1.shape[0] * image1.shape[1], 3))

# Find and display most dominant colors
cluster1 = KMeans(n_clusters=5).fit(reshape1)
visualize1 = visualize_colors(cluster1, cluster1.cluster_centers_)
visualize1 = cv2.cvtColor(visualize1, cv2.COLOR_RGB2BGR)
cv2.imshow('Dominant colors of image 1', visualize1)
cv2.waitKey()

# Load image and convert to a list of pixels
image2 = cv2.imread('output/crop.mp4/08-01-21/car2_4.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
reshape2 = image2.reshape((image2.shape[0] * image2.shape[1], 3))

# Find and display most dominant colors
cluster2 = KMeans(n_clusters=5).fit(reshape2)
visualize2 = visualize_colors(cluster2, cluster2.cluster_centers_)
visualize2 = cv2.cvtColor(visualize2, cv2.COLOR_RGB2BGR)
cv2.imshow('Dominant colors of image 2', visualize2)
cv2.waitKey()
