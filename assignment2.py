from a2_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math 
import os

### ### ### ### ### ### MY UTILITY FUNCTIONS ### ### ### ### ### ### 
allDistanceMeasures = ["L2", "ChiSquareDistance", "Intersection", "HellingerDistance"]

def formImageArrayFromPath(imagePath, numberOfBins=8):
    """
    Function returns an array of the form [imageName, image, histogram]
    for the image with path imagePath.
    """
    imageName = os.path.basename(imagePath)
    image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    histogram = myhist3(image, numberOfBins).reshape(-1)
    return [imageName, image, histogram]

def modifiedFormImageArrayFromPath(imagePath, numberOfBins=8):
    """
    Function returns an array of the form [imageName, image, histogram]
    for the image with path imagePath.

    Modified function takes into account weights of the bins using 
    simple frequency-based wighting techniques.
    """
    imageArray = formImageArrayFromPath(imagePath, numberOfBins)

    lmb = 8
    weights = [math.e**(-lmb*frequency) for frequency in imageArray[2]]
    modifiedImageHistogram = weights * imageArray[2]
    modifiedImageHistogram /= np.sum(modifiedImageHistogram)
    imageArray[2] = modifiedImageHistogram
    
    return imageArray

def formImageArray(imageName, image, numberOfBins=8):
    """
    Function returns an array of the form [imageName, image, histogram]
    for the image image with name imageName.
    """
    histogram = myhist3(image, numberOfBins).reshape(-1)
    return [imageName, image, histogram]

def modifiedFormImageArray(imageName, image, numberOfBins=8):
    """
    Function returns an array of the form [imageName, image, histogram]
    for the image image with name imageName.

    Modified function takes into account weights of the bins using 
    simple frequency-based wighting techniques.
    """
    imageArray = formImageArray(imageName, image, numberOfBins)

    lmb = 8
    weights = [math.e**(-lmb*frequency) for frequency in imageArray[2]]
    modifiedImageHistogram = weights * imageArray[2]
    modifiedImageHistogram /= np.sum(modifiedImageHistogram)
    imageArray[2] = modifiedImageHistogram
    
    return imageArray

def readAllImagesFromDir(pathToDir):
    """
    Function reads all images from the directory defined by pathToDir
    and saves them into map (key is fileName, entry is image as array) that it returns.
    Images are read in RGB color space.
    """
    images = {}    
    for file in os.listdir(pathToDir):
        image = cv2.imread(os.path.join(pathToDir, file))
        if image is not None:
            images[file] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    return images

def plotImagesAndHistograms(histogramsMap, distanceMeasure, numberOfImages):
    """
    Functions plots referenceHistogram and its numberOfImages closest histograms.
    Arrays are values in histogramsMap are of the form [imageName, image, imageHistogram].
    Keys in histogramsMap are distances from the reference image.
    Reference image is declared by the key (distance) equal to 0.
    """
    i = 1
    for distance in histogramsMap.keys():
        if (i > numberOfImages or len(histogramsMap.keys()) < i - 1):
            break

        imageArray = histogramsMap.get(distance)
        # plotting image
        plt.subplot(2, numberOfImages, i)
        plt.imshow(imageArray[1])
        plt.title("Image " + imageArray[0], fontsize=10)

        # plotting histogram
        plt.subplot(2, numberOfImages, i + numberOfImages)
        plt.plot(imageArray[2])
        plt.title(distanceMeasure + "=" + str(round(distance, 2)), fontsize=10)

        i += 1
    
    plt.suptitle("Distance measure used is " + distanceMeasure, fontsize=15)
    plt.show()

def displayMultiplePlots(plots, colors, labels=[], suptitle=""):
    """
    Function displays multiple plots on the same figure.
    """
    for plot, clr, lbl in zip(plots, colors, labels):
        shape = plot.shape
        if(len(plot.shape) > 1):
            plt.plot(plot[0], plot[1], color=clr, label=lbl)
            plt.xlim(plot[0][0], plot[0][-1])
        else:
            plt.plot(plot, color=clr, label=lbl)
    
    if(len(labels) > 0): 
        plt.legend()    
    plt.suptitle(suptitle)
    plt.show()

def displayMultiplePlotsSeparate(plots, titles, limits=[]):
    """
    Function displays multiple plots, each on one subplot in a row.
    Plots are given as elements of the array plot.
    Titles for corresponding plots are given as elements of the array titles.
    """
    i = 1
    numberOfPlots = len(plots)
    plt.rcParams["figure.autolayout"] = True
    for plot, title in zip(plots, titles):
        ax = plt.subplot(1, numberOfPlots, i)
        # if limits are defined, set them equal for every plot
        if(len(limits) > 0):
            ax.set_xlim(limits[0][0], limits[0][1])
            ax.set_ylim(limits[1][0], limits[1][1])
        ax.plot(plot)
        ax.set_title(title)        
        i += 1
    plt.show()

def displayMultipleImages(arrayOfImages, suptitle=""):
    """
    Function displays multiple images on one plot.
    Images are given in array, each element in arrayOfImages 
    is 4 element array, representing image before and after filtering
    and corresponding titles.
    Images are grayscale.
    """
    numberOfColons = len(arrayOfImages)
    i = 1
    NoneType = type(None)
    for element in arrayOfImages:
        if(type(element[0]) != NoneType):
            plt.subplot(2, numberOfColons, i)
            plt.imshow(element[0], cmap="gray")
            plt.title(element[2])

        if(type(element[1]) != NoneType):
            plt.subplot(2, numberOfColons, i + numberOfColons)
            plt.imshow(element[1], cmap="gray")
            plt.title(element[3])
        i += 1
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()

### ### ### ### ### ### EXERCISE1 ### ### ### ### ### ### 

### 1a)
def myhist3(threeChannelImage, numberOfBins=8):
    """
    Function computes a 3-D histogram from a three channel image threeChannelImage.
    The resulting histogram is stored in a 3-D matrix, the size of it 
    is determined by the parameter n_bins.  
    Histogram is normalized.
    """
    # initializing empty array for histogram
    H = np.zeros((numberOfBins, numberOfBins, numberOfBins))

    for i in range(threeChannelImage.shape[0]):
        for j in range(threeChannelImage.shape[1]):
            # getting pixel at width i and height j
            pixel = threeChannelImage[i][j]
            # scalling components of the pixel
            scalledFirstComponent = ((pixel[0] /256) * numberOfBins).astype(np.uint8)
            scalledSecondComponent = ((pixel[1] / 256) * numberOfBins).astype(np.uint8)
            scalledThirdComponent = ((pixel[2] / 256) * numberOfBins).astype(np.uint8)
            # increasing value of histogram's element at computed coordinates
            # H(R,G,B) = number of pixels with components [R,G,B]
            H[scalledFirstComponent][scalledSecondComponent][scalledThirdComponent] += 1

    # normalizing histogram
    H /= np.sum(H)
    return H

### 1b) implementing distance measures between histograms
def compare_histograms(histogram1, histogram2, distanceMeasure):
    """
    Function returns a single scalar value that represents the similarity 
    (or distance) between the two histograms, using method defined by @param distanceMeasure.
    """
    if(distanceMeasure == "L2"):
        return math.sqrt(np.sum((histogram1 - histogram2)**2))
    elif(distanceMeasure == "ChiSquareDistance"):
        return 0.5*(np.sum(((histogram1 - histogram2)**2)/(histogram1 + histogram2 + 1e-10)))
    elif(distanceMeasure == "Intersection"):
        return 1 - np.sum(np.minimum(histogram1, histogram2))
    elif(distanceMeasure == "HellingerDistance"):
        return math.sqrt(0.5*np.sum((np.sqrt(histogram1) - np.sqrt(histogram2))**2))
    else:
        return -1

### 1c)
# testing functions from 1a) and 1b)
# loading images, computing its 3D histograms and reshaping them to 1D arrays
object1Image = cv2.imread('dataset/object_01_1.png')
object2Image = cv2.imread('dataset/object_02_1.png')
object3Image = cv2.imread('dataset/object_03_1.png')

object1Histogram = myhist3(object1Image).reshape(-1)
object2Histogram = myhist3(object2Image).reshape(-1)
object3Histogram = myhist3(object3Image).reshape(-1)

# calculating other three distances between histograms using different distance measures.
for distanceMeasure in allDistanceMeasures:
    histogramsMap = {
        compare_histograms(object1Histogram, object1Histogram, distanceMeasure) : ["Object 1 image", cv2.cvtColor(object1Image, cv2.COLOR_BGR2RGB), object1Histogram],
        compare_histograms(object1Histogram, object2Histogram, distanceMeasure) : ["Object 2 image", cv2.cvtColor(object2Image, cv2.COLOR_BGR2RGB), object2Histogram],
        compare_histograms(object1Histogram, object3Histogram, distanceMeasure) : ["Object 3 image", cv2.cvtColor(object3Image, cv2.COLOR_BGR2RGB), object3Histogram]        
    }
    plotImagesAndHistograms(histogramsMap, distanceMeasure, 3)

### 1d)
def imageRetrievalSystem(pathToDir, referenceImagePath, 
    distanceMeasures=allDistanceMeasures, numberOfBins=8):
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
    referenceImageArray = formImageArrayFromPath(referenceImagePath, numberOfBins)

    # computing histograms and distances for all images and all distance measures
    for imageName in images.keys():
        image = images.get(imageName)
        imageArray = formImageArray(imageName, image, numberOfBins)
        for distanceMeasure in distanceMeasures:
            distance = compare_histograms(referenceImageArray[2], imageArray[2], distanceMeasure)
            resultMap[distanceMeasure][distance] = imageArray
    return resultMap

def displayDistancesWithHistograms(distancesMap, distanceMeasures=allDistanceMeasures):
    """
    Function displays multiple histograms (on one plot), 
    with corresponding distances from reference image, for different distanceMeasures.
    """
    for distanceMeasure in distanceMeasures:
        sortedMap = dict(sorted(distancesMap[distanceMeasure].items()))                           
        numberOfImages = min(6, len(sortedMap))
        plotImagesAndHistograms(sortedMap, distanceMeasure, numberOfImages)       

refImagePath = "dataset1d_1/object_05_4.png";
resultDistancesMap = imageRetrievalSystem("dataset1d_1", refImagePath)
displayDistancesWithHistograms(resultDistancesMap)

resultDistancesMap = imageRetrievalSystem("dataset1d_2", refImagePath)
displayDistancesWithHistograms(resultDistancesMap)


### 1e)
# displaying image distances with plots
def displayDistancesWithPlot(distancesMap, sorting=False, distanceMeasure="HellingerDistance"):
    """
    Function displays distances from a reference image on a plot for distance measure.

    @param distancesMap map of distances for some distance measures
    @param sorting determines whether the distances from reference image should be sorted out
    @param distanceMeasure specified distance measure
    """

    finalMap = {}
    if(sorting):
        finalMap = dict(sorted(distancesMap[distanceMeasure].items()))
    else:
        finalMap = dict(distancesMap.get(distanceMeasure))                           

    # on x axis plot indices of images, on y axis plot distances from reference image
    x_values = range(len(finalMap))
    y_values = np.array(list(finalMap.keys()))
    plt.plot(x_values, y_values)

    # mark 5 smallest values with red circles
    numberOfIndeces = min(5, len(y_values))
    indecesForMarkers = np.argpartition(y_values, numberOfIndeces)[0:numberOfIndeces]
    plt.plot(x_values, y_values, 'ro', markevery=indecesForMarkers, mfc='none')

    plt.show()

# unsorted image sequence
resultDistancesMap = imageRetrievalSystem("dataset", refImagePath, distanceMeasures=["HellingerDistance"], numberOfBins=8)
displayDistancesWithPlot(resultDistancesMap)

# sorted image sequence 
displayDistancesWithPlot(resultDistancesMap, True)

### 1f)
arrayOfImageArrays = list(resultDistancesMap.get("HellingerDistance").values())
imageHistograms = [imageArray[2] for imageArray in arrayOfImageArrays]

summedHistograms = sum(imageHistograms)
displayMultiplePlotsSeparate([summedHistograms], ["Summing all image (from dataset) histograms bin-wise"])

def modifiedImageRetrievalSystem(pathToDir, referenceImagePath, 
    distanceMeasures=allDistanceMeasures, numberOfBins=8):
    """
    @param pathToDir path to directory with all images
    @param referenceImagePath path to the reference image
    @param distanceMeasures
    @param numberOfBins for histograms.

    @return resultMap is a dict, where keys are distanceMeasures, and values are
    dictonaries. Keys of that dictonaries are distances of individual image to the reference image, 
    while values are arrays of the form [imageName, image, imageHistogram].

    Modified image retrieval system takes into account weights of the bins using 
    simple frequency-based wighting techniques.
    """    
    # initializing map of maps for different distance measures.
    resultMap = {distanceMeasure:{} for distanceMeasure in distanceMeasures}

    # loading images from directory and reference image
    images = readAllImagesFromDir(pathToDir)

    # form an array [imageName, image, histogram] for referenceImage
    referenceImageArray = modifiedFormImageArrayFromPath(referenceImagePath)

    # computing histograms and distances for all images and all distance measures
    for imageName in images.keys():
        image = images.get(imageName)
        imageArray = modifiedFormImageArray(imageName, image)
        for distanceMeasure in distanceMeasures:
            distance = compare_histograms(referenceImageArray[2], imageArray[2], distanceMeasure)
            resultMap[distanceMeasure][distance] = imageArray
    return resultMap

resultDistancesMap = modifiedImageRetrievalSystem("dataset1d_1", refImagePath)
displayDistancesWithHistograms(resultDistancesMap)

"""
Report your observations. Did the weighting help with retrieving relevant results?
Answer: 
    - Histograms with applied weighting are more equally distributed.
    - Retrieved sequencies are the same as with the simplified function.
    - Calculated distances are generally larger compared to calculated with simplified function.
"""

### ### ### ### ### ### EXERCISE2 ### ### ### ### ### ### 

### 2a)
# manually computed 

### 2b)
def simple_convolution(signal, kernel):
    """
    Function uses a 1-D signal and a kernel of size 2N + 1
    and returns the convolution between the two. 
    Convolutaion on signal elements is calculated 
    from i = N to i = |signal| - N.
    """
    N = int((len(kernel) - 1) / 2)
    result = np.zeros((len(signal) - 2*N))

    flippedKernel = np.flip(kernel)
    for i in range (N, len(signal) - N):
        result[i - N] = np.sum(flippedKernel*signal[(i-N):(i+N+1)])

    return result

# loading signal and kernel and computing their convolution
signal = read_data("signal.txt")
kernel = read_data("kernel.txt")
convolution = simple_convolution(signal, kernel)

# displaying signal, kernel and result 
displayMultiplePlots([signal, kernel, convolution, np.array(cv2.filter2D(signal, -1, kernel)).reshape(-1)], 
            ["blue", "orange", "green", "red"], ["Original", "Kernel", "Result", "cv2"], 
            "Original signal, kernel and result after convolution,\ncompared to result got with cv2.filter2D")

### 2c)
def modified_simple_convolution(signal, kernel):
    """
    Modified function simple_convolution, it addresses also the 
    edges of the signal. Method used is to set output values to 0.   
    """
    # copy signal array and extend it to address also edges of the signal
    N = int((len(kernel) - 1) / 2)
    signalCopy = np.concatenate((np.zeros((1, N)), signal.copy(), np.zeros((1, N))), axis=None)
    return simple_convolution(signalCopy, kernel)

# displaying signal, kernel and result 
convolution = modified_simple_convolution(signal, kernel)
displayMultiplePlots([signal, kernel, convolution],
        ["blue", "orange", "green"], ["Original", "Kernel", "Result"], 
        "Result of convolution got with modified_simple_convolution")

### 2d)
# Gaussian filtering
def gaussianKernel(sigma=1):
    """
    Function calculates a normalized Gaussian kernel
    and returns list where 1st element is x values, 2nd is computed kernel. 
    Size of the kernel is 2*[3*sigma] + 1    
    """
    x_values = range(int(-np.ceil(3*sigma)), int(np.ceil(3*sigma) + 1))
    kernel = []
    for x in x_values:
        kernel.append((1/(math.sqrt(2*math.pi)*sigma))*(math.e**(-(x**2)/(2*sigma**2))))

    # normalizing kernel
    kernel /= np.sum(kernel)    
    return np.array([x_values, kernel])

sigmas = [0.5, 1, 2, 3, 4]
gaussianKernels = [gaussianKernel(sigma) for sigma in sigmas]  
gaussianKernelsLabels = ["sigma=" + str(sigma) for sigma in sigmas]
gaussianKernelsColors = ["blue", "orange", "green", "red", "purple"]

displayMultiplePlots(gaussianKernels, gaussianKernelsColors, gaussianKernelsLabels, "Gaussian kernels for different sigmas")

### 2e)
# Filtering with gaussian kernel k1 with sigma=2 and then with predefined kernel k2
k1 = gaussianKernel(2)[1]
k2 = [0.1, 0.6, 0.4]
firstFiltering = np.convolve(np.convolve(signal, k1), k2)

# Filtering with with predefined kernel k2 and then with gaussian kernel k1 with sigma=2
secondFiltering = np.convolve(np.convolve(signal, k2), k1)

# Filtering with kernel k1*k2
thirdFiltering = np.convolve(signal, np.convolve(k1, k2))

displayMultiplePlotsSeparate([signal, firstFiltering, secondFiltering, thirdFiltering], ["s", "(s*k1)*k2", "(s*k2)*k1", "s*(k1*k2)"]) 

### ### ### ### ### ### EXERCISE3 ### ### ### ### ### ### 

### 3a)
def gaussfilter(image, sigma=1):
    """ 
    Function generates a Gaussian filter and applies it to 2D image. 
    @return filtered image.
    """
    kernel = gaussianKernel(sigma)[1]
    filteredImage = cv2.filter2D(image, -1, kernel)
    kernel = np.array([kernel]).T
    filteredImage = cv2.filter2D(filteredImage, -1, kernel)
    return filteredImage

# loading image and converting it to grayscale
lenaGSImage = cv2.cvtColor(cv2.imread("images/lena.png"), cv2.COLOR_BGR2GRAY)

# applying Gaussian noise to the image and filtering it using Gaussian filter
lenaGaussianNoise = gauss_noise(lenaGSImage)
lenaGaussianNoiseFiltered = gaussfilter(lenaGaussianNoise, 2)

# applying SP noise to the image and filtering it using Gaussian filter0
lenaSPNoise = sp_noise(lenaGSImage)
lenaSPNoiseFiltered = gaussfilter(lenaSPNoise, 2)

displayMultipleImages([[lenaGSImage, None, "Original", ""], 
    [lenaGaussianNoise, lenaGaussianNoiseFiltered, "Gaussian Noise", "Filtered Gaussian Noise"], 
    [lenaSPNoise, lenaSPNoiseFiltered, "Salt and Pepper Noise", "Filtered Salt and Peper Noise"]], 
    "Comparison of Gaussian filtering on Gaussian and SP noise")

### 3b) 
# convolution for image sharpening - linear filters
museumGSImage = cv2.cvtColor(cv2.imread("images/museum.jpg"), cv2.COLOR_BGR2GRAY)
kernel = [[0, 0, 0], [0, 2, 0], [0, 0, 0]] - np.ones((3, 3))/9
linearyFiltered = cv2.filter2D(museumGSImage, -1, kernel)
displayMultipleImages([[museumGSImage, linearyFiltered, "Original", "Applying linear filter to sharpen image"]], "Linear filter")

### 3c) 
# nonlinear filter - median filter
def simple_median(signal, filterWidth=5):
    """
    Function takes input signal and filter width and returns
    the filtered signal using median filter.
    Note: filterWidth = 2k + 1
    """
    k = int((filterWidth - 1) / 2)
    filteredSignal = []
    for i in range(len(signal)):
        filterWindow = signal[max(0, i - k):min(len(signal), i + k + 1)]
        filteredSignal.append(np.median(filterWindow))
    return filteredSignal

# loading signal and applying SP noise to the signal
signal2 = read_data("signal2.txt")
spNoiseSignal = signal2.copy()
spNoiseSignal[np.random.rand(signal2.shape[0]) < 0.07] = 3
spNoiseSignal[np.random.rand(signal2.shape[0]) < 0.07] = 0

# filtering image with median and Gaussian filter
gaussianFilteredSignal = gaussfilter(spNoiseSignal)
medianFilteredSignal = simple_median(spNoiseSignal)

displayMultiplePlotsSeparate([signal2, spNoiseSignal, gaussianFilteredSignal, medianFilteredSignal],
    ["Original", "Corrupted", "Gauss", "Median"], [[0, 40], [0, 3.5]])

### 3d)
# 2D median filter
def medianfilter2D(image, filterDimensions=[3, 3]):
    """
    Function takes input image and filter dimensions and returns
    the filtered signal using median filter.
    
    Note: filterDimensions=[filterWidth, filterHeight], where
    filterWidth = 2*k1 + 1 and filterHeight = 2*k2 + 1
    """
    filteredImage = image.copy()
    k1 = int((filterDimensions[0] - 1) / 2)
    k2 = int((filterDimensions[1] - 1) / 2)
    
    imageWidth, imageHeight = image.shape
    # going through all pixels of the image and computing filteredImage
    for i in range(imageWidth):
        for j in range(imageHeight):
            # defining neighborhood window
            lowerXLimit = max(0, i - k1)
            upperXLimit = min(imageWidth, i + k1 + 1)
            lowerYLimit = max(0, j - k2)
            upperYLimit = min(imageHeight, j + k2 + 1)

            filterWindow = image[lowerXLimit:upperXLimit, lowerYLimit:upperYLimit]
            filteredImage[i, j] = np.median(filterWindow)
    return filteredImage

lenaGaussianNoiseMedianFiltered = medianfilter2D(lenaGaussianNoise)
lenaSPNoiseMedianFiltered = medianfilter2D(lenaSPNoise)

displayMultipleImages([[lenaGSImage, None, "Original", ""], 
    [lenaGaussianNoise, lenaSPNoise, "Gaussian Noise", "Salt and Pepper Noise"], 
    [lenaGaussianNoiseFiltered, lenaSPNoiseFiltered, "Gauss filtered", "Gauss filtered"],
    [lenaGaussianNoiseMedianFiltered, lenaSPNoiseMedianFiltered, "Median filtered", "Median filtered"]], suptitle="Comparision between Gauss and Median filtering")

### 3e) hybrid image merging
def laplacianFilter(image, sigma=1):
    """
    Function returns image filtered with approximation of 
    Laplacian filter.
    """
    x_values, gaussKernel = gaussianKernel(sigma)
    unitImpulse = np.zeros((len(x_values)))
    unitImpulse[int(len(x_values) / 2)] = 1
    laplacianKernel = unitImpulse - gaussKernel

    filteredImage = image.copy().astype(np.float64)
    filteredImage = cv2.filter2D(filteredImage, -1, laplacianKernel)
    laplacianKernel = np.array([laplacianKernel]).T
    filteredImage = cv2.filter2D(filteredImage, -1, laplacianKernel)

    return filteredImage

# loading image and filtering it with Gaussian filter
lincolnImage = cv2.cvtColor(cv2.imread("images/lincoln.jpg"), cv2.COLOR_BGR2GRAY)
filteredLincolnImage = gaussfilter(lincolnImage, sigma=5)

# loading image and filtering it with Laplacian filter
obamaImage = cv2.cvtColor(cv2.imread("images/obama.jpg"), cv2.COLOR_BGR2GRAY)

filteredObamaImage = laplacianFilter(obamaImage, 30)

mergedImage = ((0.35*filteredLincolnImage + 0.65*filteredObamaImage).astype(np.float64))
displayMultipleImages([[lincolnImage, filteredLincolnImage, "Original Lincoln image", "Filtered Lincoln image with Gaussian"], 
[obamaImage, filteredObamaImage, "Original Obama image", "Filtered Obama image with Laplacian"], [mergedImage, None, "Result", ""]])
