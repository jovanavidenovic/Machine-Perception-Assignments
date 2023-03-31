import numpy as np
import cv2 
from matplotlib import pyplot as plt
import math
import random
import a6_utils
import os
from matplotlib import animation as animation


### ### ### MY UTILITY FUNCTIONS ### ### ###
def displayMultipleImages_InOneRow(arrayOfImages, colorMap="", suptitle=""):
    """
    Function displays multiple images on one plot.
    Images are given in array, each element in arrayOfImages is 2 element array, 
    representing image and corresponding title. 
    @param colorMap defines the color map used in plt.imshow. 
        If it is "", default color map is applied.
    """
    numberOfColons = len(arrayOfImages)
    i = 1
    for element in arrayOfImages:
        plt.subplot(1, numberOfColons, i)
        if(colorMap != ""):
            plt.imshow(element[0], cmap=colorMap)
        else:
            plt.imshow(element[0])
        plt.title(element[1], fontsize=8)
        i += 1
    
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()

def displayMultiplePlots(arrayOfPlots, suptitle=""):
    """
    Function displays multiple plots on one plot.
    Data to be ploted is given in array, each element in arrayOfPlots is 2 element array, 
    representing list of the form [x_valuesForPlot, y_valuesForPlot] and corresponding title. 
    """
    numberOfColons = int(np.ceil(len(arrayOfPlots) / 2))
    i = 1
    for element in arrayOfPlots:
        plt.subplot(2, numberOfColons, i)
        plt.plot(element[0][0], element[0][1])
        plt.title(element[1])
        i += 1
    
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()

def displayPoints(points, show=True):
    for point in points:
        plt.scatter(point[0], point[1])
    if(show):
        plt.show()


### ### ### EXERCISE 1 ### ### ###
### 1b) PCA algorithm
def pca(points):
    """ 
    @param points containing of N rows, each contains x and y
    coordinate of the point.
    Implementation of the direct PCA algorithm.
    """
    # centering the points
    X = np.array(points).T
    mean = np.mean(X, axis=1)
    Xd = [point - mean for point in points]
    Xd = np.array(Xd).T

    # calculating covariance matrix and its eigenvectors
    C = np.cov(Xd)
    U, S, _ = np.linalg.svd(C, full_matrices=True)
    projectedPoints = []
    for point in points:
        projectedPoints.append(np.dot(U.T, point - mean))
    return (mean, U, S, C, projectedPoints)



### 1b) PCA algorithm - example
points = np.loadtxt("data/points.txt")
mean, U, S, covarianceMatrix, projectedPoints = pca(points)
print(U)


### 1c) plotting points, ellipse and eigenvectors
displayPoints(points, False)
a6_utils.drawEllipse(mean, covarianceMatrix, n_std=1)
i = 0
UT = U.copy().T
for eigvector in UT:
    eigvector *= np.sqrt(S[i])
    color = "red"
    if(i % 2 == 1):
        color = "green"
    plt.quiver(*mean, eigvector[0], eigvector[1], color=color, scale=8.5)
    i += 1
plt.xlim([-1, 7])
plt.ylim([-1, 7])
plt.title("Direct PCA")
plt.show()


### 1d) normalized cumulative graph of the eigenvalues
def normalize(values, lowerBound = 0, upperBound = 1):
    minValue = min(values)
    maxValue = max(values)
    return [lowerBound + (x - minValue) * (upperBound - lowerBound) / (maxValue - minValue) for x in values]

cumulativeSum = []
cumulativeSum.append(0)
for eigenvalue in S:
    cumulativeSum.append(cumulativeSum[-1] + eigenvalue)

cumulativeSum = normalize(cumulativeSum)
print(cumulativeSum)
plt.plot(cumulativeSum)
plt.title("Normalized cumulative sum of eigenvalues")
plt.show()

# 83.6% of the variance is explained with using just the first eigenvalue 


#### 1e) projecting points back to Cartesian space with diminished matrix U
U_diminished = []
U_diminished.append(U[:, 0])
U_diminished.append([0, 0])
U_diminished = np.array(U_diminished)

reprojectedPoints = []
for projectedPoint in projectedPoints:
    reprojectedPoints.append(np.dot(U_diminished, projectedPoint) + mean)

plt.title("Projecting points back to Cartesian space with diminished matrix U");
displayPoints(reprojectedPoints)


# Data is projected to the line.


### calculate the closest point to the targetPoint from points using Euclidean distance
def findClosestPoint(points, targetPoint):
    numberOfPoints = len(points)
    distance = np.zeros((numberOfPoints))
    for i in range(numberOfPoints):
        distance[i] = np.sqrt((points[i][0] - targetPoint[0])**2 + (points[i][1] - targetPoint[1])**2)

    indexMin = np.argmin(distance)
    return (indexMin, points[indexMin])


### 1f)
q = np.array([[6], [6]])
indexClosest, closestPoint = findClosestPoint(points, q)
q = q.T
print("Closest point to [6, 6] is " + str(closestPoint) + ". That is point number " + str(int(indexClosest)) + ".")

points_wq = np.concatenate([points, q], axis=0)

mean_wq, U_wq, S_wq, covarianceMatrix_wq, projectedPoints_wq = pca(points_wq)

U_wq_diminished = []
U_wq_diminished.append(U_wq[:, 0])
U_wq_diminished.append([0, 0])
U_wq_diminished = np.array(U_wq_diminished)

reprojectedPoints_wq = []
for projectedPoint in projectedPoints_wq:
    reprojectedPoints_wq.append(np.dot(U_wq_diminished, projectedPoint) + mean_wq)

plt.title("Projecting points back to Cartesian space with diminished matrix U");
displayPoints(reprojectedPoints_wq)
reprojectedPoints_wq = np.array(reprojectedPoints_wq)
reprojected_q = reprojectedPoints_wq[-1, :]
indexClosest_rp, closestPoint_rp = findClosestPoint(reprojectedPoints_wq[:-1, :], reprojected_q)
print("Closest point to reprojected q is " + str(closestPoint_rp)  + ". That is point number " + str(int(indexClosest_rp)) + ".")


### ### ### EXERCISE 2 ### ### ###
### 2a) Dual PCA algorithm
def dual_pca(points):
    """ 
    @param points containing of N rows, each contains x and y
    coordinate of the point.
    Implementation of the dual PCA algorithm.
    """
    # number of points
    N = len(points)

    # centering the points
    X = np.array(points).T
    mean = np.mean(X, axis=1)

    Xd = [point - mean for point in points]
    Xd = np.array(Xd).T
        
    # calculating covariance matrix and its eigenvectors
    C = (1/(N-1))*np.dot(Xd.T, Xd)
    
    U, S, _ = np.linalg.svd(C, full_matrices=True)
    U = np.dot(points.T, U) * np.sqrt(1 / ((S + 1e-15)*(N-1)))
    U = U[:, :2]
    projectedPoints = []
    for point in points:
        projectedPoints.append(np.dot(U.T, point - mean))
    return (mean, U, S, C, projectedPoints)


### 2a) dual PCA - example
points_dualPCA = points.copy()
mean_dualPCA, U_dualPCA, S_dualPCA, covarianceMatrix_dualPCA, projectedPoints_dualPCA = dual_pca(points_dualPCA)
print(U_dualPCA)
print(S_dualPCA)


reprojectedPoints_dualPCA = []
for projectedPoint in projectedPoints_dualPCA:
    reprojectedPoints_dualPCA.append(np.dot(U_dualPCA, projectedPoint) + mean_dualPCA)

plt.suptitle("Dual PCA")
plt.subplot(1, 2, 1)
plt.title("Original points")
displayPoints(points, show=False)

plt.subplot(1, 2, 2)
plt.title("Reprojected points")
displayPoints(reprojectedPoints_dualPCA)



### ### ### EXERCISE 3 ### ### ###
### 3a) data preparation
def loadImages(dir):
    data = np.zeros((8064, 64))
    i = 0
    for file in os.listdir(dir):
        image = cv2.cvtColor(cv2.imread(os.path.join(dir, file)), cv2.COLOR_BGR2GRAY)
        image = image.reshape(-1)
        data[:, i] = image
        i += 1
    return data



transformedImageToMatrix = loadImages("data/faces/1")


### 3b) Dual PCA algorithm for matrix of image vectors as input
def dual_pca_images(points):
    """ 
    Implementation of the dual PCA algorithm for matrix of image vectors as input.
    """
    # number of points
    (M, N) = points.shape
    data = points.copy()
    # centering the points
    mean = np.mean(data, axis = 1)

    for i in range(N):
        data[:, i] = data[:, i] - mean

    # calculating covariance matrix and its eigenvectors
    C = (1/(N-1))*np.dot(data.T, data)
    
    U, S, _ = np.linalg.svd(C)
    U = np.dot(points, U) * np.sqrt(1 / (S*(N-1)))
    U[:, -1] = 1e-9
    return (U, S, mean)


### 3b) dual PCA on images
(U_img, S_img, mean_img) = dual_pca_images(transformedImageToMatrix)


### 3b)
for i in range(5):
    plt.subplot(1, 5, i + 1)
    eigenvector = U_img.copy()[:, i]
    eigenvector = eigenvector.reshape(96, 84)
    plt.title(str(i + 1) + ". eigenvector", fontsize=8)
    plt.imshow(eigenvector, cmap="gray")
plt.show()


### 3b)
firstImage = cv2.cvtColor(cv2.imread("data/faces/1/001.png"), cv2.COLOR_BGR2GRAY)
(height, width) = firstImage.shape

processedImage = firstImage.copy().reshape(-1)
projectedImage = np.dot(U_img.copy().T, (processedImage - mean_img))
reprojectedImage = np.dot(U_img.copy(), projectedImage) + mean_img

plt.subplot(1, 3, 1)
plt.title("Original image", fontsize = 8)
plt.imshow(reprojectedImage.reshape(height, width), cmap="gray")

processedImage_changedOriginal = firstImage.copy().reshape(-1)
processedImage_changedOriginal[4074] = 0

plt.subplot(1, 3, 2)
plt.title("Changed component\nin orginal space", fontsize = 8)
plt.imshow(processedImage_changedOriginal.reshape(height, width), cmap="gray")

projectedImage_changedPCA = projectedImage.copy()
projectedImage_changedPCA[0] = 0
reprojectedImage_changedPCA = np.dot(U_img.copy(), projectedImage_changedPCA) + mean_img
plt.subplot(1, 3, 3)
plt.title("Changed component\nin PCA space", fontsize = 8)
plt.imshow(reprojectedImage_changedPCA.reshape(height, width), cmap="gray")

plt.show()


### 3c) effect of the number of components on the reconstruction
firstImage = cv2.cvtColor(cv2.imread("data/faces/1/001.png"), cv2.COLOR_BGR2GRAY)
(height, width) = firstImage.shape
i = 0
for numberOfEigenvectors in [32, 16, 8, 4, 2, 1]:
    U_img_changed = U_img.copy()
    U_img_changed[:, numberOfEigenvectors:-1] = 0

    processedImage = firstImage.copy().reshape(-1)
    projectedImage = np.dot(U_img_changed.T, (processedImage - mean_img))
    reprojectedImage = np.dot(U_img_changed, projectedImage) + mean_img

    plt.subplot(1, 6, i + 1)
    plt.title(str(numberOfEigenvectors), fontsize = 8)
    plt.imshow(reprojectedImage.reshape(height, width), cmap="gray")
    i += 1

plt.show()


### 3d) 
images2 = loadImages("data/faces/2")
(U_img2, S_img2, mean_img2) = dual_pca_images(images2)

average_projected = np.dot(U_img2.T, mean_img2)
animatedImages = []

factor1 = np.sin(np.linspace(-10, 10))*3000
factor2 = np.cos(np.linspace(-10, 10))*3000

for i in range(20):
    U_img2_changed = U_img2.copy()
    average_projected[0] = factor1[i]
    average_projected[3] = factor2[i]
    reprojectedImage2 = np.dot(U_img2_changed, average_projected)
    # im.set_data(reprojectedImage2.reshape((height, width)))
    plt.subplot(4, 5, i + 1)
    plt.imshow(reprojectedImage2.reshape((height, width)), cmap="gray")
    # plt.pause(1)

plt.show()


### 3e) Reconstruction of a foreign image
elephantImage = cv2.cvtColor(cv2.imread("data/elephant.jpg"), cv2.COLOR_BGR2GRAY)

processedImage_elephant = elephantImage.copy().reshape(-1)
projectedImage_elephant = np.dot(U_img.copy().T, (processedImage_elephant - mean_img))
reprojectedImage_elephant = np.dot(U_img.copy(), projectedImage_elephant) + mean_img

plt.subplot(1, 2, 1)
plt.title("Original elephant image", fontsize = 8)
plt.imshow(elephantImage, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Reconstructed elephant image", fontsize = 8)
plt.imshow(reprojectedImage_elephant.reshape(height, width), cmap="gray")

plt.show()

dataPCA = np.zeros((8064, 64*3))
dataPCA[:, 0:64] = loadImages("data/faces/1")
dataPCA[:, 64:128] = loadImages("data/faces/2")
dataPCA[:, 128:192] = loadImages("data/faces/3")

dataPCA = (dataPCA - np.min(dataPCA))/(np.max(dataPCA) - np.min(dataPCA))

U, S, mean = dual_pca_images(dataPCA.copy())
mean = mean.reshape((-1,1))
dataPCA = U.T @ (dataPCA - mean)

plt.subplot(1, 2, 1)
for i in range(64):
    plt.scatter(dataPCA[0, i], dataPCA[1, i], color="yellow")
for i in range(64):
    plt.scatter(dataPCA[0, 64 + i], dataPCA[1, 64 + i], color="purple")
for i in range(64):
    plt.scatter(dataPCA[0, 128 + i], dataPCA[1, 128 + i], color="blue")
plt.title("PCA")

mean = np.mean(dataPCA, axis=1, keepdims=True)

dataPCACopy = dataPCA[:10, ]
SB = 0
SW = 0
meanAll = np.mean(dataPCACopy, axis=1, keepdims=True)
meanAll = meanAll.reshape(10, 1)
for i in range(3):
    meanClass = np.average(dataPCACopy[:, i * 64: (i + 1) * 64], axis=1)
    meanClass = meanClass.reshape(10, 1)
    SB += 64 * (meanClass-meanAll) @ (meanClass-meanAll).T
    for j in range(64):
        SW += (dataPCACopy[:,[i * 64 + j]].reshape(10, 1) - meanClass) @ (dataPCACopy[:,[i *  64 + j]].reshape(10, 1) - meanClass).T
[U_LDA, _, _] = np.linalg.svd(np.linalg.inv(SW) @ SB)

dataLDA = U_LDA.T @ (dataPCA - mean)[:10]
plt.subplot(1,2,2)
for i in range(64):
    plt.scatter(dataLDA[0, i], dataLDA[1, i], color="yellow")
for i in range(64):
    plt.scatter(dataLDA[0, 64 + i], dataLDA[1, 64 + i], color="purple")
for i in range(64):
    plt.scatter(dataLDA[0, 128 + i], dataLDA[1, 128 + i], color="blue")
plt.title("LDA")
plt.show()


