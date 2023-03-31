from a3_utils import *
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2
from assignment2_functions import *


### ### ### ### ### ### MY UTILITY FUNCTIONS ### ### ### ### ### ### 
def gaussianKernel(sigma=1) -> list:
    """
    Function calculates a normalized 1D Gaussian kernel and returns it. 
    Length of the list representing the derivative is 2*[3*sigma] + 1).
    """
    x_values = range(int(-np.ceil(3*sigma)), int(np.ceil(3*sigma) + 1))
    kernel = []
    for x in x_values:
        kernel.append(1/(math.sqrt(2*math.pi)*sigma)*np.exp(-(x**2)/(2*sigma**2)))
        
    # normalizing kernel
    kernel /= np.sum(kernel)    
    return kernel

def displayMultipleImages(arrayOfImages, suptitle=""):
    """
    Function displays multiple images on one plot.
    Images are given in array, each element in arrayOfImages is 2 element array, 
    representing image and corresponding title. Images are grayscale.
    """
    numberOfColons = int(np.ceil(len(arrayOfImages) / 2))
    i = 1
    for element in arrayOfImages:
        plt.subplot(2, numberOfColons, i)
        plt.imshow(element[0], cmap="gray")
        plt.title(element[1])
        i += 1
    
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()

def displayMultipleImages_InOneRow(arrayOfImages, suptitle=""):
    """
    Function displays multiple images on one plot in one row.
    Images are given in array, each element in arrayOfImages is 2 element array, 
    representing image and corresponding title. Images are grayscale.
    """
    numberOfColons = len(arrayOfImages)
    i = 1
    for element in arrayOfImages:
        plt.subplot(1, numberOfColons, i)
        plt.imshow(element[0])
        plt.title(element[1])
        i += 1
    
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()


### ### ### ### ### ### EXERCISE1 ### ### ### ### ### ### 

### 1a)
# Look at writtenTasks/derivatives.pdf


### 1b) computing derivative of a 1D Gaussian kernel
def gaussdx(sigma = 1) -> list:
    """
    Function calculates a normalized derivative of 1D Gaussian kernel
    and returns it. 
    Length of the list representing the derivative is 2*[3*sigma] + 1).
    """
    x_values = range(int(-np.ceil(3*sigma)), int(np.ceil(3*sigma) + 1))
    derivative = []
    for x in x_values:
        derivative.append(-1/(math.sqrt(2*math.pi)*sigma**3)*x*np.exp(-x**2/(2*sigma**2)))

    # normalizing derivative
    derivative /= np.sum(np.abs(derivative))   
    return derivative


### 1c) impulse response function
# discrete version of Dirac function
impulse = np.zeros((50, 50))
impulse[25, 25] = 1

# generating Gaussian 1D kernel and its derivative
G = np.array([gaussianKernel(sigma=5)])
GT = G.T
D = np.array([gaussdx(sigma=5)])
# cv2.filter2D is implemented with correlation -> need to flip non-symetric kernels
D = np.flip(D, axis=1)
DT = D.T

i = 1
# (a) First convolution with G and then convolution with GT
imgGGT = cv2.filter2D(cv2.filter2D(impulse, -1, G), -1, GT)
# (b) First convolution with G and then convolution with DT .
imgGDT = cv2.filter2D(cv2.filter2D(impulse, -1, G), -1, DT)
# (c) First convolution with D and then convolution with GT .
imgDGT = cv2.filter2D(cv2.filter2D(impulse, -1, D), -1, GT)
# (d) First convolution with GT and then convolution with D.
imgGTD = cv2.filter2D(cv2.filter2D(impulse, -1, GT), -1, D)
# (e) First convolution with DT and then convolution with G.
imgDTG = cv2.filter2D(cv2.filter2D(impulse, -1, DT), -1, G)

displayMultipleImages([[impulse, "Impulse"], [imgGDT, "G, DT"], [imgDGT, "D, GT"], 
    [imgGGT, "G, GT"], [imgGTD, "GT, D"], [imgDTG, "DT, G"]])


### 1d) computing partial derivatives
def gaussianKernelsAndPartialDerivatives(sigma=1) -> list:
    """
    Function returns list of the Gaussian kernels and 
    its first order derivatives in the form [GX, GY, DX, DY]. 
    Both kernels and derivatives are flipped so they can be used directly in cv2.filter2D
    """
    GX = np.array([gaussianKernel(sigma)])
    GX = np.flip(GX, axis=1)
    GY = GX.T
    DX = np.array([gaussdx(sigma)])
    DX = np.flip(DX, axis=1)
    DY = DX.T   
    return [GX, GY, DX, DY]

def partialDerivatives_firstOrder(image, sigma=1) -> list:
    """
    Function computes first order partial derivatives of an image 
    to respect to x and y. 
    """
    image = image.astype(np.float64)
    # computing Gaussian kernels and partial derivatives.
    [GX, GY, DX, DY] = gaussianKernelsAndPartialDerivatives(sigma)
    
    image_dx = cv2.filter2D(cv2.filter2D(image, -1, GY), -1, DX)
    image_dy = cv2.filter2D(cv2.filter2D(image, -1, GX), -1, DY)

    return [image_dx, image_dy]

def partialDerivatives_secondOrder(image, sigma=1) -> list:
    """
    Function computes second order partial derivatives of an image.
    """
    # computing Gaussian kernels and partial derivatives.
    [GX, GY, DX, DY] = gaussianKernelsAndPartialDerivatives(sigma)    
    [image_dx, image_dy] = partialDerivatives_firstOrder(image, sigma)

    image_dxx = cv2.filter2D(cv2.filter2D(image_dx, -1, GY), -1, DX)
    image_dxy = cv2.filter2D(cv2.filter2D(image_dx, -1, GX), -1, DY)
    image_dyx = image_dxy.copy()
    image_dyy = cv2.filter2D(cv2.filter2D(image_dy, -1, GX), -1, DY)

    return [image_dxx, image_dxy, image_dyx, image_dyy]

def gradient_magnitude(image, sigma=1) -> list:
    """
    Function returns derivatve magnitudes and angles of the image.
    """
    [image_dx, image_dy] = partialDerivatives_firstOrder(image, sigma)
    image_magnitudes = np.sqrt(image_dx**2 + image_dy**2)
    image_angles = np.arctan2(image_dy, image_dx)  

    return [image_magnitudes, image_angles]

# loading image as grayscale
museumImage = cv2.cvtColor(cv2.imread("images/museum.jpg"), cv2.COLOR_BGR2GRAY)
# normalizing pixels of the image to be between 0 and 1
museumImage = museumImage.astype(np.float64)
museumImage *= 1/255

[museumImage_dx, museumImage_dy] = partialDerivatives_firstOrder(museumImage, sigma=1)
[museumImage_dxx, museumImage_dxy, museumImage_dyx, museumImage_dyy] = partialDerivatives_secondOrder(museumImage, sigma=0.5)
[museumImage_magnitudes, museumImage_angles] = gradient_magnitude(museumImage, sigma=1)

displayMultipleImages([[museumImage, "Original"], [museumImage_dx, "l_x"], [museumImage_dy, "l_y"], [museumImage_magnitudes, "l_mag"], 
[museumImage_dxx, "l_xx"], [museumImage_dxy, "l_xy"], [museumImage_dyy, "l_yy"], [museumImage_angles, "l_dir"]])



### 1e) extension to image retrieval system
def formImageArrayFromPath(imagePath, sigma):
    """
    Function returns an array of the form [imageName, image, histogram]
    for the image with path imagePath.
    Image is read as grayscale.
    """
    imageName = os.path.basename(imagePath)
    image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
    histogram = getHistogramBasedOnGradient(image, sigma)
    return [imageName, image, histogram]

def formImageArray(imageName, image, sigma):
    """
    Function returns an array of the form [imageName, image, histogram]
    for the image image with name imageName.
    """
    histogram = getHistogramBasedOnGradient(image, sigma)
    return [imageName, image, histogram]
    
def histogramOfMagnitudeToAngle(image_magnitudes, image_angles):
    """
    Function computes histogram of gradient magnitudes with respect to 
    gradient angles by:
        - converting angles to degrees and quantitizing them to 8 values
        - for each pixel of the cell, adds the value of the gradient to 
          the bin specified by the corresponding angle 
    """
    image_angles_copy = image_angles.copy()
    image_angles_copy *= 180 / np.pi
    image_angles_copy += 180
    image_angles_copy *= 7/360
    (height, width) = image_magnitudes.shape

    histogram = np.zeros((8, ))
    for i in range(height):
        for j in range(width):
            histogram[int(image_angles_copy[i][j])] += image_magnitudes[i][j]
    
    return histogram

def getHistogramBasedOnGradient(image, sigma=1):
    """
    Function:
    - computes gradient magnitudes / angles for entire image
    - divides the image in 8 x 8 grid
    - compute histogram for each grid
    - concatenate all histograms
    - normalize result histogram
    """
    [image_magnitudes, image_angles] = gradient_magnitude(image, sigma)
    (height, width) = image.shape
    cellHeight = int(np.floor(height / 8))
    cellWidth = int(np.floor(width / 8))

    cells_coordinates = []
    for i in range(8):
        for j in range(8):
            cells_coordinates.append([i*cellHeight, min(i*cellHeight + cellHeight, height), j*cellWidth, min(j*cellWidth + cellWidth, width)])

    histograms = []
    for coordinates in cells_coordinates:
        cell_magnitudes = image_magnitudes[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
        cell_angles = image_angles[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]]
        histograms.extend(histogramOfMagnitudeToAngle(cell_magnitudes, cell_angles))

    histograms /= np.sum(histograms)
    return histograms


def imageRetrievalSystem(pathToDir, referenceImagePath, 
    distanceMeasures=["HellingerDistance"], numberOfBins=8):
    """
    @param pathToDir path to directory with all images
    @param referenceImagePath path to the reference image
    @param distanceMeasures
    @param numberOfBins for histograms.

    @return resultMap is a dict, where keys are distanceMeasures, and values are
    dictonaries. Keys of that dictonaries are distances of individual image to the reference image, 
    while values are arrays of the form [imageName, image, imageHistogram].
    """    
    # initializing map of maps for different distance measures.
    resultMap = {distanceMeasure:{} for distanceMeasure in distanceMeasures}

    # loading images from directory and reference image
    images = readAllImagesFromDir(pathToDir)

    # form an array [imageName, image, histogram] for referenceImage
    referenceImageArray = formImageArrayFromPath(referenceImagePath, sigma=1)

    # computing histograms and distances for all images and all distance measures
    for imageName in images.keys():
        image = images.get(imageName)
        imageArray = formImageArray(imageName, image, sigma=1)
        for distanceMeasure in distanceMeasures:
            distance = compare_histograms(referenceImageArray[2], imageArray[2], distanceMeasure)
            resultMap[distanceMeasure][distance] = imageArray
    return resultMap

refImagePath = "dataset/object_05_4.png"
resultDistancesMap = imageRetrievalSystem("dataset", refImagePath, ["HellingerDistance"])
displayDistancesWithHistograms(resultDistancesMap, ["HellingerDistance"])


### ### ### ### ### ### EXERCISE2 ### ### ### ### ### ### 
### 2a)
def findedges(image, sigma, theta):
    """
    Function creates a binary matrix image_e, that only keeps 
    pixels with magnitude higher than threshold theta.
    """
    image_magnitudes = gradient_magnitude(image, sigma)[0]
    (height, width) = image.shape
    resultImage = image.copy()
    for i in range(height):
        for j in range(width):
            resultImage[i][j] = 1 if image_magnitudes[i][j] >= theta else 0
    return resultImage

museumImage_thresholded = findedges(museumImage, 1, 0.16)


### 2b)
def greaterThanNeighborhood(image_angles, image_magnitudes, center_i, center_j) -> bool:
    """
    Function checks the neighboring pixels parallel to the gradient direction and
    returns false if it is not the largest in the neighborhood (based on derivative 
    magnitude), otherwise returns true.
    """
    (height, width) = image_magnitudes.shape
    referenceMagnitude = image_magnitudes[center_i][center_j]
    
    # finding neighbors based on gradient angle
    gradientAngle = image_angles[center_i][center_j] * 180 / np.pi
    if(gradientAngle < 0):
        gradientAngle += 180

    if (0 <= gradientAngle < 22.5 or 157.5 <= gradientAngle <= 180):
        dir_i = 0
        dir_j = 1
    elif (22.5 <= gradientAngle < 67.5):
        dir_i = 1
        dir_j = 1
    elif (67.5 <= gradientAngle < 112.5):
        dir_i = 0
        dir_j = 1
    else:
        dir_i = -1
        dir_j = 1

    neighbor1_i = center_i + dir_i
    neighbor1_j = center_j + dir_j
    
    neighbor2_i = center_i - dir_i
    neighbor2_j = center_j - dir_j

    if((0 <= neighbor1_i and neighbor1_i < height and 0 <= neighbor1_j and neighbor1_j < width and image_magnitudes[neighbor1_i][neighbor1_j] >= referenceMagnitude) or 
        (0 <= neighbor2_i and neighbor2_i < height and 0 <= neighbor2_j and neighbor2_j < width and image_magnitudes[neighbor2_i][neighbor2_j] >= referenceMagnitude)):  
        return False

    return True


def nonMaximaSuppression(image, sigma, theta):
    """
    Function applies non-maxima suppression on image and returns result.
    Sigma and theta are used for finding inital edges using function findedges and 
    for computing gradient magnitudes and angles.
    """
    [image_magnitudes, image_angles] = gradient_magnitude(image, sigma)
    nmsImage = findedges(image, sigma, theta)
    (height, width) = image.shape

    for i in range(height):
        for j in range(width):
            if not greaterThanNeighborhood(image_angles, image_magnitudes, i, j):
                nmsImage[i][j] = 0 
    
    return nmsImage

museumImage_nms = nonMaximaSuppression(museumImage, 1, 0.16)


### 2c) edge tracking by hysteresis
def edgeTrackingByHysteresis(image, sigma, tLow, tHigh):
    """
    Function creates a binary matrix image_e, where are kept pixels 
    with magnitude higher than tHigh, discarded pixels with 
    magnitude lower than tLow and decided about the rest based 
    on connectivity with pixels with magnitude above tHigh.
    """
    # apply non-maxima suppression with threshold tLow
    image_magnitudes = gradient_magnitude(image, sigma)[0]
    resultImage = nonMaximaSuppression(image, sigma, tLow).astype(np.uint8)
    # connect contours using 4-connectedness    
    connectivity = 8
    output = cv2.connectedComponentsWithStats(resultImage, connectivity, cv2.CV_32S)
    numLabels = output[0]
    labels = output[1]
    # trace each contour separately 
    for label in range(1, numLabels):
        indices = labels == label
        exists = image_magnitudes[indices] > tHigh
        if True in exists:
            resultImage[labels == label] = 1
        else:
            resultImage[labels == label] = 0
    return resultImage

museumImage_hysteresis = edgeTrackingByHysteresis(museumImage, sigma=1, tLow=0.04, tHigh=0.12)
displayMultipleImages([[museumImage, "Original"], [museumImage_thresholded, "Thresholded image"], 
[museumImage_nms, "NMS of image"], [museumImage_hysteresis, "Hysteresis"]])


### ### ### ### ### ### EXERCISE3 ### ### ### ### ### ### 
### 3a) calculating accumulator matrix
def addToAccumulatorMatrix(accumulatorMatrix, x, y):
    """
    Function increases value of the element corresponding 
    to x and y in the accumulator matrix.
    """
    numberOfBins_d = accumulatorMatrix.shape[0]
    numberOfBins_angle = accumulatorMatrix.shape[0]
    linspace_angle = np.linspace(-np.pi/2, np.pi, numberOfBins_angle)
    for i in range(numberOfBins_angle):
        angle = linspace_angle[i]
        d = x*math.cos(angle) + y*math.sin(angle) + numberOfBins_d/2
        if(d >= 0 and d <= numberOfBins_d):
            accumulatorMatrix[int(d)][i] += 1
    return accumulatorMatrix

# calculating sinusoids in Hough space for different points in (x, y) space
accumulatorMatrix = np.zeros((300, 300))
accumulatorMatrix_x10y10 = addToAccumulatorMatrix(accumulatorMatrix.copy(), 10, 10)
accumulatorMatrix_x30y60 = addToAccumulatorMatrix(accumulatorMatrix.copy(), 30, 60)
accumulatorMatrix_x50y20 = addToAccumulatorMatrix(accumulatorMatrix.copy(), 50, 20)
accumulatorMatrix_x80y90 = addToAccumulatorMatrix(accumulatorMatrix.copy(), 80, 90)

displayMultipleImages([[accumulatorMatrix_x10y10, "x=10, y=10"], [accumulatorMatrix_x30y60, "x=30, y=60"],
[accumulatorMatrix_x50y20, "x=50, y=20"], [accumulatorMatrix_x80y90, "x=80, y=90"]])


### 3b) 
def findClosest(d, linspace_d):
    """
    Function returns the index of the closest element to the d in the linspace_d.
    """
    difference_array = np.absolute(linspace_d-d)
    return difference_array.argmin()

def hough_find_lines(binaryImage, numberOfBins_d, numberOfBins_angle, threshold):
    """
    Function generates a curve in the (angle, d) space by using equation 
    xcos(angle) + ysin(angle) = d, for each nonzero pixel in the image,
    for all possible values of angle and increases the corresponding cells in 
    accumulator matrix. 
    @return generated accumulator matrix.
    """
    (height, width) = binaryImage.shape
    diagonal = math.sqrt(height**2 + width**2)
    
    accumulatorMatrix = np.zeros((numberOfBins_d, numberOfBins_angle))
    linspace_angle = np.linspace(-np.pi/2, np.pi/2, numberOfBins_angle)
    for i in range(height):
        for j in range(width):
            if(binaryImage[i][j] != 0):
                for index_angle in range(len(linspace_angle)):
                    angle = linspace_angle[index_angle]
                    d = j*math.cos(angle) + i*math.sin(angle)
                    index_d = np.round(((d + diagonal)/(2*diagonal)) * numberOfBins_d).astype(int)
                    accumulatorMatrix[index_d][index_angle] += 1
    return accumulatorMatrix

# testing function on different images
synteticImage = np.zeros((100, 100))
synteticImage[10][10] = 1
synteticImage[10][20] = 1
findLines_synteticImage = hough_find_lines(synteticImage, 200, 200, 0)

onelineImage = cv2.cvtColor(cv2.imread("images/oneline.png"), cv2.COLOR_BGR2GRAY)
onelineImage_edges = findedges(onelineImage, 1, 0.5)
findLines_onelineImage = hough_find_lines(onelineImage_edges, 200, 200, 0)

rectangleImage = cv2.cvtColor(cv2.imread("images/rectangle.png"), cv2.COLOR_BGR2GRAY)
rectangleImage_edges = findedges(rectangleImage, 1, 0.5)
findLines_rectangleImage = hough_find_lines(rectangleImage_edges, 200, 200, 0)

displayMultipleImages_InOneRow([[findLines_synteticImage, "Synthetic"], 
[findLines_onelineImage, "oneline.png"], [findLines_rectangleImage, "rectangle.png"]], 
suptitle="Testing function hough_find_lines")



### 3c)
def greaterThan8Neighborhood(image, center_i, center_j) -> bool:
    """
    Function checks the neighboring pixels and
    returns false if it is not the largest in the 8-neighborhood, otherwise returns true.
    """
    (height, width) = image.shape
    referenceValue = image[center_i][center_j]
    
    # going through neighbors
    for i in range(max(0, center_i - 1), min(height, center_i + 2)):
        for j in range(max(0, center_j - 1), min(width, center_j + 2)):
            if(image[i][j] > referenceValue or 
            (image[i][j] == referenceValue and (i < center_i or (i == center_i and j < center_j)))):
                return False
    return True

def nonmaxima_suppression_box(gsImage):
    """
    Function checks the neighborhood of each pixel and set it to 0 if it is 
    not the maximum value in the neighborhood (only consider 8-neighborhood).
    @return image with applied non-maxima suppresion.
    """
    
    resultImage = gsImage.copy()
    (height, width) = resultImage.shape
    for i in range(height):
        for j in range(width):
               if(not greaterThan8Neighborhood(gsImage, i, j)): 
                    resultImage[i][j] = 0
    return resultImage

nms_synteticImage = nonmaxima_suppression_box(findLines_synteticImage)
nms_oneLineImage = nonmaxima_suppression_box(findLines_onelineImage)
nms_rectangleImage = nonmaxima_suppression_box(findLines_rectangleImage)

displayMultipleImages_InOneRow([[nms_oneLineImage, "oneline.png"], 
[nms_rectangleImage, "rectangle.png"]], suptitle="Testing function nonmaxima_suppression_box")


### 3d)
def draw_line(imageForDrawing, d, angle):
    """
    Function draws a line with polar coordinates d and angle
    on the image given with @param imageForDrawing.
    """
    if(len(imageForDrawing.shape) == 2):
        (height, width) = imageForDrawing.shape
    else:
        (height, width, _) = imageForDrawing.shape   
    x1 = 0
    y1 = int((d - x1*math.cos(angle))/math.sin(angle))
    x2 = width
    y2 = int((d - x2*math.cos(angle))/math.sin(angle))
    cv2.line(imageForDrawing, (x1, y1), (x2, y2), (36, 97, 0), 2)

def extractParameters(image, accumulatorMatrix, threshold, numberOfBins_d, numberOfBins_angle):
    """
    Function extracts all the parameter pairs (d, angle) whose
    corresponding accumulator cell value is greater than @param threshold.
    @return dictonary, where key is number of votes and value is list of 
    the form [d, angle].
    """
    (height_am, width_am) = accumulatorMatrix.shape
    if(len(image.shape) == 2):
        (height_image, width_image) = image.shape
    else:
        (height_image, width_image, _) = image.shape

    diagonal = math.sqrt(height_image**2 + width_image**2)
    linspace_angle = np.linspace(-np.pi/2, np.pi/2, numberOfBins_angle)
    linspace_d = np.linspace(0, 2*diagonal, numberOfBins_d)
    
    result = {}
    for d_i in range(height_am):
        for angle_i in range(width_am):
            if(accumulatorMatrix[d_i][angle_i] >= threshold):
                angle = linspace_angle[angle_i]
                d = linspace_d[d_i] - diagonal
                result[accumulatorMatrix[d_i][angle_i]] = [d, angle]
    return result

def drawLinesOnImage(imageForDrawing, params):
    """
    Helper function to draw lines determined by params on the image.
    """
    if(len(params) > 10):
        params = dict(sorted(params.items()))
        params = {K:V for (K,V) in [param for param in params.items()][-10:]}
    for param in params.values():
        param = list(param)
        draw_line(imageForDrawing, d=param[0], angle=param[1])
    plt.imshow(imageForDrawing)
    plt.title("Extracting line parameters with largest values\nin accumulator matrix")
    plt.show()

params_onelineImage = extractParameters(onelineImage, nms_oneLineImage, 1000, 200, 200)
drawLinesOnImage(onelineImage, params_onelineImage)
params_rectangleImage = extractParameters(rectangleImage, nms_rectangleImage, 500, 200, 200)
drawLinesOnImage(rectangleImage, params_rectangleImage)


### 3e)
# loading images
bricksImage_rgb = cv2.cvtColor(cv2.imread("images/bricks.jpg"), cv2.COLOR_BGR2RGB)
pierImage_rgb = cv2.cvtColor(cv2.imread("images/pier.jpg"), cv2.COLOR_BGR2RGB)

bricksImage = cv2.cvtColor(cv2.imread("images/bricks.jpg"), cv2.COLOR_BGR2GRAY)
pierImage = cv2.cvtColor(cv2.imread("images/pier.jpg"), cv2.COLOR_BGR2GRAY)

# normalizing images
bricksImage = bricksImage.astype(np.float64)
bricksImage *= 1/255
pierImage = pierImage.astype(np.float64)
pierImage *= 1/255

# detecting edges
findEdges_bricksImage = findedges(bricksImage, sigma=1.5, theta=0.1)
findEdges_pierImage = findedges(pierImage, sigma=1.25, theta=0.12)

# detecting lines
findLines_bricksImage = hough_find_lines(findEdges_bricksImage, 200, 200, 0)
findLines_pierImage = hough_find_lines(findEdges_pierImage, 200, 200, 0)
displayMultipleImages_InOneRow([[findLines_bricksImage, "Finding lines on bricks.jpg"], [findLines_pierImage, "Finding lines on pier.jpg"]])
nms_bricksImage = nonmaxima_suppression_box(findLines_bricksImage)
nms_pierImage = nonmaxima_suppression_box(findLines_pierImage)

params_bricksImage = extractParameters(bricksImage_rgb, nms_bricksImage, threshold=400, numberOfBins_d=200, numberOfBins_angle=200)
drawLinesOnImage(bricksImage_rgb, params_bricksImage)
params_pierImage = extractParameters(pierImage_rgb, nms_pierImage, threshold=300, numberOfBins_d=200, numberOfBins_angle=200)
drawLinesOnImage(pierImage_rgb, params_pierImage)


### 3f) Modified Hough transform, uses gradient angle
def modified_hough_find_lines(originalImage, binaryImage, numberOfBins_d, numberOfBins_angle, threshold):
    """
    Function generates a curve in the (angle, d) space by using equation 
    xcos(angle) + ysin(angle) = d, for each nonzero pixel in the image,
    for angle equal to local gradient angle and increases the corresponding cells in 
    accumulator matrix. 
    @return generated accumulator matrix.
    """
    (height, width) = binaryImage.shape
    diagonal = math.sqrt(height**2 + width**2)
    
    accumulatorMatrix = np.zeros((numberOfBins_d, numberOfBins_angle))
    linspace_angle = np.linspace(-np.pi/2, np.pi/2, numberOfBins_angle)
    
    image_magnitudes, image_angles = gradient_magnitude(originalImage, sigma = 1)

    # normalizing angles to be between -np.pi/2 and np.pi/2
    image_angles_copy = image_angles.copy()
    image_angles_copy[image_angles < (- np.pi / 2)] +=  np.pi
    image_angles_copy[image_angles > (np.pi / 2)] -= np.pi

    for i in range(height):
        for j in range(width):
            if(binaryImage[i][j] != 0):
                angle = image_angles_copy[i][j]
                index_angle = findClosest(angle, linspace_angle)
                d = j*math.cos(angle) + i*math.sin(angle)
                index_d = np.round(((d + diagonal)/(2*diagonal)) * numberOfBins_d).astype(int)
                accumulatorMatrix[index_d][index_angle] += image_magnitudes[i][j]
    return accumulatorMatrix

    


### 3e) with modified function
# loading images
bricksImage_rgb = cv2.cvtColor(cv2.imread("images/bricks.jpg"), cv2.COLOR_BGR2RGB)
pierImage_rgb = cv2.cvtColor(cv2.imread("images/pier.jpg"), cv2.COLOR_BGR2RGB)

bricksImage = cv2.cvtColor(cv2.imread("images/bricks.jpg"), cv2.COLOR_BGR2GRAY)
pierImage = cv2.cvtColor(cv2.imread("images/pier.jpg"), cv2.COLOR_BGR2GRAY)

# normalizing images
bricksImage = bricksImage.astype(np.float64)
bricksImage *= 1/255
pierImage = pierImage.astype(np.float64)
pierImage *= 1/255

# detecting edges
findEdges_bricksImage = findedges(bricksImage, sigma=1.5, theta=0.1)
findEdges_pierImage = findedges(pierImage, sigma=1.25, theta=0.1)
# detecting lines
findLines_bricksImage = modified_hough_find_lines(bricksImage, findEdges_bricksImage, 200, 200, 0)
findLines_pierImage = modified_hough_find_lines(pierImage, findEdges_pierImage, 200, 200, 0)

nms_bricksImage = nonmaxima_suppression_box(findLines_bricksImage)
nms_pierImage = nonmaxima_suppression_box(findLines_pierImage)

params_bricksImage = extractParameters(bricksImage_rgb, nms_bricksImage, threshold=1, numberOfBins_d=200, numberOfBins_angle=200)
drawLinesOnImage(bricksImage_rgb, params_bricksImage)
params_pierImage = extractParameters(pierImage_rgb, nms_pierImage, threshold=1, numberOfBins_d=200, numberOfBins_angle=200)
drawLinesOnImage(pierImage_rgb, params_pierImage)

### 3e) with modified function
# loading images
rectangleImage_rgb = cv2.cvtColor(cv2.imread("images/rectangle.png"), cv2.COLOR_BGR2RGB)
rectangleImage = cv2.cvtColor(cv2.imread("images/rectangle.png"), cv2.COLOR_BGR2GRAY)

# normalizing images
rectangleImage = rectangleImage.astype(np.float64)
rectangleImage *= 1/255

# detecting edges
findEdges_rectangleImage = findedges(rectangleImage, sigma=1, theta=0.3)
#findEdges_rectangleImage = cv2.Canny(rectangleImage, threshold1=0.04, threshold2=0.06)
# detecting lines
findLines_rectangleImage = modified_hough_find_lines(rectangleImage, findEdges_rectangleImage, 200, 200, 0)
nms_rectangleImage = nonmaxima_suppression_box(findLines_rectangleImage)
params_rectangleImage = extractParameters(rectangleImage_rgb, nms_rectangleImage, threshold=1, numberOfBins_d=200, numberOfBins_angle=200)
drawLinesOnImage(rectangleImage_rgb, params_rectangleImage)

### 3g) Hough transform that detects circles with fixed radius
eclipseImage = cv2.cvtColor(cv2.imread("images/eclipse.jpg"), cv2.COLOR_BGR2GRAY)
eclipseImage_rgb = cv2.cvtColor(cv2.imread("images/eclipse.jpg"), cv2.COLOR_BGR2RGB)

# normalizing image
eclipseImage = eclipseImage.astype(np.float64)
eclipseImage *= 1/255

def draw_circle(imageForDrawing, centerCoordinates, r):
    """
    Function draws a line with polar coordinates d and angle
    on the image given with @param imageForDrawing.
    """
    if(len(imageForDrawing.shape) == 2):
        (height, width) = imageForDrawing.shape
    else:
        (height, width, _) = imageForDrawing.shape   
    cv2.circle(imageForDrawing, centerCoordinates, r, (36, 97, 0), 2)

def drawCirclesOnImage(imageForDrawing, list_centerCoordinates, r):
    """
    Helper function to draw lines determined by params on the image.
    """
    for centerCoordinates in list_centerCoordinates:
        draw_circle(imageForDrawing, centerCoordinates, r)

def hough_find_circles(binaryImage, radius):
    """
    Function implements Hough transform for detecting circles of a fixed radius.
    Radius is given, function is creating accumaltor matrix for searching for centre coordinates.
    @return generated accumulator matrix.
    """
    (height, width) = binaryImage.shape
    image_magnitudes, image_angles = gradient_magnitude(binaryImage, sigma=1)
    
    # normalizing angles to be between 0 and np.pi
    image_angles_copy = image_angles.copy()
    image_angles_copy[image_angles < - np.pi] +=  np.pi
    image_angles_copy[image_angles > np.pi] -= np.pi

    accumulatorMatrix = np.zeros((height, width))
    #angle_linspace = np.linspace(0, 2*np.pi, 360)
    for y in range(height):
        for x in range(width):
            if(binaryImage[y][x] != 0):
                for angle in [image_angles_copy[y][x]]:
                    a = int(x + radius*math.cos(angle))
                    b = int(y + radius*math.sin(angle))
                    if(a > 0 and a < width and  b > 0 and b < height):
                        accumulatorMatrix[b][a] += image_magnitudes[y][x]
    return accumulatorMatrix

def extractParametersForCircle(accumulatorMatrix, threshold):
    """
    Function extracts all the parameter pairs (d, angle) whose
    corresponding accumulator cell value is greater than @param threshold.
    @return dictonary, where key is number of votes and value is list of 
    the form [d, angle].
    """
    (height_am, width_am) = accumulatorMatrix.shape
    result = []
    for b in range(height_am):
        for a in range(width_am):
            if(accumulatorMatrix[b][a] >= threshold):
                result.append((a, b))             
    return result

# detecting edges
findEdges_eclipseImage = edgeTrackingByHysteresis(eclipseImage, sigma=1, tLow=0.05, tHigh=0.1)
drawingImage = findEdges_eclipseImage.copy()
drawingImage *= 255
thresholds = [2.9, 2.9, 3.6, 3.3, 3.6, 3.25]
t_i = 0
for radius in range(45, 51):
    findCircles_eclipseImage = hough_find_circles(findEdges_eclipseImage, radius)
    nms_eclipseImage = nonmaxima_suppression_box(findCircles_eclipseImage)
    
    # displayMultipleImages([[accumulatorMatrix, "accumulator matrix for radius " + str(radius)]])
    list_centerCoordinates = extractParametersForCircle(nms_eclipseImage, 0.63)
    t_i += 1
    drawCirclesOnImage(eclipseImage_rgb, list_centerCoordinates, radius)

displayMultipleImages([[eclipseImage_rgb, "Hough transform for detecting circles of a fixed radius"]])



