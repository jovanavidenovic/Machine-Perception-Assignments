import numpy as np
import cv2 
from matplotlib import pyplot as plt
import math
import random
import a4_utils
import a5_utils


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

def harris_detector(gsImage, threshold, sigma=1, sigma1=1.6, alfa=0.06):
    """
    Implementation of Harris corner detector.

    @param image grayscale image for which the corners should be detected.
    @param threshold threshold.
    @param sigma standard deviation used in Gaussian filtering during derivations of the image.
    @param sigma1 standard deviation used in Gaussian filtering, used as weight function.

    @return list of the form [gsImage, cornerResponseFunction, cornerResponseFunction_nms, title] 
    """
    [image_dx, image_dy] = partialDerivatives_firstOrder(gsImage, sigma)

    # matrix M used in Harris corner detector is [[M11, M12], [M12, M22]].
    M11 = cv2.GaussianBlur(image_dx**2, (0, 0), sigma1)
    M12 = cv2.GaussianBlur(image_dx*image_dy, (0, 0), sigma1)
    M22 = cv2.GaussianBlur(image_dy**2, (0, 0), sigma1)

    # forming corner response function and applying threshold    
    detM = M11*M22 - M12**2
    traceM = M11 + M22
    cornerResponseFunction = detM - alfa*((traceM)**2)
    cornerResponseFunction_thresholded = cornerResponseFunction.copy()
    cornerResponseFunction_thresholded[cornerResponseFunction_thresholded <= threshold] = 0

    # performing non-maximum suppression
    cornerResponseFunction_nms = nonmaxima_suppression_box(cornerResponseFunction_thresholded)
    return [gsImage, cornerResponseFunction, cornerResponseFunction_nms, "sigma=" + str(sigma)]

def find_correspondences(desc1, desc2):
    """
    Function calculattes similarities between all descriptors in @param desc1 and
    @param desc2, and for each descriptor from the first lists, finds the most similar
    descriptor from the second list. Uses Hellinger distance.

    @return a list of [a, b] pairs, where a is the index from the first list, and b is 
    the index from the second list.
    """ 
    correspondences = []
    for index_firstDesc in range(len(desc1)):
        d_1 = desc1[index_firstDesc]
        similarities = [math.sqrt(0.5*np.sum((np.sqrt(d_1) - np.sqrt(d_2))**2)) for d_2 in desc2]
        index_secondDesc = similarities.index(min(similarities))
        correspondences.append(np.array([index_firstDesc, index_secondDesc]))
    return np.array(correspondences)

def simple_matching(image1, image2, descriptor="simple"):
    """ 
    Follow the algorithm below:
        - Execute a feature point detector to get stable points for both images.
        - Compute simple descriptors for all detected feature points.
        - Find best matches between descriptors in left and right images using the
          Hellinger distance, i.e. compute the best matches from the left to right image
          and then the other way around. 
        - Only select symmetric matches.
    @return list [pts1, pts2], where elements represents the coordinates of points of the symmetric matches.
    """
    # finding coordinates of feature points using Harris feature point detector.
    featurePointsDetector1 = harris_detector(image1, threshold=1e-6, sigma=27, sigma1=37.8)
    fPCoords1 = np.nonzero(featurePointsDetector1[2])

    featurePointsDetector2 = harris_detector(image2, threshold=1e-6, sigma=27, sigma1=37.8)
    fPCoords2 = np.nonzero(featurePointsDetector2[2])

    if(descriptor == "simple"):
      # computing simple descriptors
      desc1 = a4_utils.simple_descriptors(image1, fPCoords1[0], fPCoords1[1])
      desc2 = a4_utils.simple_descriptors(image2, fPCoords2[0], fPCoords2[1])
      
    # finding correspondences from the first to the second image
    correspondences1 = find_correspondences(desc1, desc2)

    # finding correspondences from the second to the first image
    correspondences2 = find_correspondences(desc2, desc1)

    # select symmetric matches
    symmetricCorrespondences = []

    for corr in correspondences1:    
      flippedCorr = np.flip(corr)
      is_symmetric = np.any(np.all(flippedCorr == correspondences2, axis=1))
      if(is_symmetric):
        symmetricCorrespondences.append(corr)

    symmetricCorrespondences = np.array(symmetricCorrespondences)

    # calculating coordinates of the correspondences
    pts1 = [[fPCoords1[1][index], fPCoords1[0][index]] for index in symmetricCorrespondences[:, 0]]
    pts2 = [[fPCoords2[1][index], fPCoords2[0][index]] for index in symmetricCorrespondences[:, 1]] 

    return [pts1, pts2]

def modified_find_correspondences(desc1, desc2, threshold, epsilon = 1e-6):
    """
    Function 
    It calculattes similarities between all descriptors in @param desc1 and
    @param desc2, and for each descriptor from the first lists, finds the most similar
    and second most similar descriptor from the second list and computes its ratio. 
    If ratio is bigger than the @param threshold, adds that correspondence to the result.
    Uses Hellinger distance. 

    @return a list of [a, b] pairs, where a is the index from the first list, and b is 
    the index from the second list.
    """ 
    correspondences = []
    for index_firstDesc in range(len(desc1)):
        d_1 = desc1[index_firstDesc]
        similarities = [math.sqrt(0.5*np.sum((np.sqrt(d_1) - np.sqrt(d_2))**2)) for d_2 in desc2]
        index_secondDesc = similarities.index(min(similarities))
        index_secondMax_secondDesc = similarities.index(sorted(similarities)[1])
        if(similarities[index_secondDesc] / (similarities[index_secondMax_secondDesc] + epsilon) < threshold):
          correspondences.append(np.array([index_firstDesc, index_secondDesc]))
    return np.array(correspondences)

def modified_matching(image1, image2, descriptor="simple", threshold=0.8):
    """ 
    Function is updated version on simple_matching function.
    It calculates the distance between the point in the first image and the second-most similar 
    keypoint and the most similar keypoint in the second image. Ratio (first/second) is low for distinctive
    key points and high for non-disctinctive ones. To keep only non-distinctive key matches, we apply a threshold,
    defined by @param threshold.
    @return list [filtered_pts1, filtered_pts2], where elements represents the filtered coordinates of points of the matches.
    """
    # finding coordinates of feature points using Harris feature point detector.
    featurePointsDetector1 = harris_detector(image1, threshold=1e-6, sigma=9, sigma1=12.6)
    fPCoords1 = np.nonzero(featurePointsDetector1[2])

    featurePointsDetector2 = harris_detector(image2, threshold=1e-6, sigma=9, sigma1=12.6)
    fPCoords2 = np.nonzero(featurePointsDetector2[2])

    if(descriptor == "simple"):
      # computing simple descriptors
      desc1 = a4_utils.simple_descriptors(image1, fPCoords1[0], fPCoords1[1])
      desc2 = a4_utils.simple_descriptors(image2, fPCoords2[0], fPCoords2[1])

    # finding correspondences from the first to the second image
    correspondences1 = modified_find_correspondences(desc1, desc2, threshold)

    # finding correspondences from the second to the first image
    correspondences2 = modified_find_correspondences(desc2, desc1, threshold)

    # select symmetric matches
    symmetricCorrespondences = []

    for corr in correspondences1:    
      flippedCorr = np.flip(corr)
      is_symmetric = np.any(np.all(flippedCorr == correspondences2, axis=1))
      if(is_symmetric):
        symmetricCorrespondences.append(corr)

    symmetricCorrespondences = np.array(symmetricCorrespondences)

    # calculating coordinates of the correspondences
    pts1 = [[fPCoords1[1][index], fPCoords1[0][index]] for index in symmetricCorrespondences[:, 0]]
    pts2 = [[fPCoords2[1][index], fPCoords2[0][index]] for index in symmetricCorrespondences[:, 1]] 

    pts1_choosen = []
    pts2_choosen = []
    for i in range(0, len(pts1), 4):
        pts1_choosen.append(pts1[i])
        pts2_choosen.append(pts2[i])

    return [pts1_choosen, pts2_choosen]

# 2D median filter
def medianfilter2D(image, filterDimensions=[35, 35]):
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


### ### ### EXERCISE 1 ### ### ###
### 1b) compute the disparity.
def computeDisparity(f, T, pz):
    """ 
    Function computes the disparity for the predefined values of 
    focal length and baseline displacement between cameras for
    a range of values of depths.
    @param f focal length.
    @param T baseline displacement between cameras.
    @param pz list of values of depths.
    
    @return list of computed disparities.
    """
    disparities = f*T/pz
    return disparities


### 1b) compute the disparity for predefined data - example.
f = 2.5
T = 120
pz = np.array(range(100, 1000, 2))
disparities = computeDisparity(f, T, pz)
displayMultiplePlots([[[pz, disparities], "Calculating disparities for given range of depths;\nfocal length and baseline displacement were predefined"]])


### 1d) calculate disparity for an image pair.
def normalizedCrossCorrelation(X, Y):
    """ 
    Returns normalized cross correlation, calculated on matrices X and Y.
    """
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    if(X.shape != Y.shape):
        return -10000
        
    numerator = np.sum([(x_i - mean_X)*(y_i - mean_Y) for x_i, y_i in zip(X, Y)])
    denumerator = math.sqrt(np.sum([(x_i - mean_X) ** 2 for x_i in X])*np.sum([(y_i - mean_Y) ** 2 for y_i in Y]))
    if(denumerator == 0):
        return 0
    ncc = numerator / denumerator
    return ncc

def calculateDisparity(image1, image2, patchSize=5):
    (height, width) = image1.shape
    disparityMap = np.zeros((height, width))

    for y in range(0, height, patchSize):
        for x in range(0, width, patchSize):
            image1_patch = image1[y:(min(height, y + patchSize)), x:(min(width, x + patchSize))]
            (height_ps, width_ps) = image1_patch.shape
            changed = False
            resNCC = 0
            currentDisparity = 0
            for x_image2 in range(0, width - width_ps):    
                image2_patch = image2[y:(min(height, y + height_ps)), x_image2:(min(width, x_image2 + width_ps))]
                currentNCC = normalizedCrossCorrelation(image1_patch, image2_patch)
                if(currentNCC > resNCC or not changed):
                    changed = True
                    resNCC = currentNCC
                    currentDisparity = x_image2 - x
            disparityMap[y:min(height, (y + height_ps)), x:min(width, (x + width_ps))] = currentDisparity
    return disparityMap


def createImageWithDisparity(image, disparityMap):
    """ 
    Function to compute the second image in stereo system, 
    using the original image and disparity map.
    """
    (height, width) = image.shape
    resImage = image.copy()

    for i in range(height):
        for j in range(width):
            j_res = int(j + disparityMap[i][j])
            resImage[i][min(width - 1, j_res)] = image[i][j]
    return resImage


### 1d) calculate disparity for an image pair - example.
image_office_left = cv2.cvtColor(cv2.imread("data/disparity/office_left.png"), cv2.COLOR_BGR2GRAY)
(height, width) = image_office_left.shape
image_office_left = cv2.resize(image_office_left, (int(width/2), int(height/2)))
image_office_right = cv2.cvtColor(cv2.imread("data/disparity/office_right.png"), cv2.COLOR_BGR2GRAY)
image_office_right = cv2.resize(image_office_right, (int(width/2), int(height/2)))

office_disparityMap = calculateDisparity(image_office_left, image_office_right)


projectedImage = createImageWithDisparity(image_office_left, office_disparityMap)
displayMultipleImages_InOneRow([[office_disparityMap, "Creating disparity map\nfor an image pair"], [image_office_left, "Image office_left.png"], 
[projectedImage, "Computed image\nusing disparity map"], [image_office_right, "Image office_right.png"]], colorMap="gray")


### 1e) modified disparity map
def calculateDisparity_modified(image1, image2, patchSize=5, windowSize=35):
    disparityMap1 = calculateDisparity(image1, image2)
    disparityMap2 = calculateDisparity(image2, image1)

    (height, width) = image1.shape
    merged_disparityMap = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            j1 = int(j + disparityMap1[i, j])
            j2 = j1 + disparityMap2[i, j1]
            merged_disparityMap[i][j] = int(0.5*(j1 - j2))

    merged_disparityMap = medianfilter2D(merged_disparityMap)
    return merged_disparityMap


### 1e) modified disparity map - example.
office_disparityMap_modified = calculateDisparity_modified(image_office_left, image_office_right)

projectedImage_modified = createImageWithDisparity(image_office_left, office_disparityMap_modified)
displayMultipleImages_InOneRow([[office_disparityMap_modified, "Creating disparity map for\nan image pair - modified"], [image_office_left, "Image office_left.png"], 
[projectedImage_modified, "Computed image\nusing disparity map"], [image_office_right, "Image office_right.png"]], colorMap="gray")


# function for drawing epipoles
def drawEpipoles(pairOfImages, pairOfCorrespondences, F):
    [image_left, image_right] = pairOfImages
    [points_leftImage, points_rightImage] = pairOfCorrespondences
    # left image
    plt.subplot(1, 2, 1)
    (height_left, width_left) = image_left.shape
    plt.imshow(image_left, cmap="gray")
    for point in points_rightImage:
        point = np.append(point, [1])
        l = np.dot(F.T, point)
        a5_utils.draw_epiline(l, height_left, width_left)
    for point in points_leftImage:
        plt.scatter(point[0], point[1], s=15, c='red', marker='o')
    
    # right image
    plt.subplot(1, 2, 2)
    (height_right, width_right) = image_right.shape
    plt.imshow(image_right, cmap="gray")
    for point in points_leftImage:
        point = np.append(point, 1)
        l = np.dot(F, point)
        a5_utils.draw_epiline(l, height_right, width_right)
    for point in points_rightImage:
        plt.scatter(point[0], point[1], s=15, c='red', marker='o')
    plt.show()


### ### ### EXERCISE 2 ### ### ###
### 2b) function for estimating fundamental matrix
def fundamental_matrix(pairsOfPoints, showImages=False, pairOfImages=[]):
    """ 
    Function estimates fundamental matrix using 8-point algorithm.
    Applies preconditioning and after construction, reconstructs F so   
    it has a rank equal to 2. Then removes the preconditioning and returns 
    calculated F.
    @param pairsOfPoints array of the form [[Nx2 array of points of left image], [Nx2 array of points of right image]]
    @param showImages determines whether the calculated epipolar lines should be displayed.
    @param pairOfImages array of the form [image1, image2]
    """

    # apply preconditioning
    points_leftImage = np.array(pairsOfPoints[0])
    (normalizedPoints_leftImage, T_left) = a5_utils.normalize_points(points_leftImage)  
    
    x_l = normalizedPoints_leftImage[:, 0]
    y_l = normalizedPoints_leftImage[:, 1]

    points_rightImage = np.array(pairsOfPoints[1])
    (normalizedPoints_rightImage, T_right) = a5_utils.normalize_points(points_rightImage)
    
    x_r = normalizedPoints_rightImage[:, 0] 
    y_r = normalizedPoints_rightImage[:, 1] 

    numberOfPoints = len(x_l)

    # 8-point algorithm
    A = np.zeros((numberOfPoints, 9))
    for i in range(numberOfPoints):
        u = x_l[i]
        v = y_l[i]
        u_ = x_r[i]
        v_ = y_r[i]
        
        A[i] = np.array([u*u_, u_*v, u_, u*v_, v*v_, v_, u, v, 1])

    _, _, VT = np.linalg.svd(A)
    V = VT.T

    # compute vector f and reshape it to 3x3
    f = V[:, -1]
    F_ = np.reshape(f, (3, 3))

    # reconstrucing matrix F so it has rank 2
    U, D, VT = np.linalg.svd(F_)
    D_expanded = np.zeros((3, 3))
    D_expanded[0, 0] = D[0]
    D_expanded[1, 1] = D[1]

    F_ = np.dot(U, np.dot(D_expanded, VT))

    # removing preconditioning
    F = np.dot(np.dot(T_right.T, F_), T_left)
    
    if(showImages):
        # draw epipoles
        drawEpipoles(pairOfImages, pairsOfPoints, F)

    return F


### 2b) function for estimating fundamental matrix - example
house_image1 = cv2.cvtColor(cv2.imread("data/epipolar/house1.jpg"), cv2.COLOR_BGR2GRAY)
house_image2 = cv2.cvtColor(cv2.imread("data/epipolar/house2.jpg"), cv2.COLOR_BGR2GRAY)

correspondencePairs = np.loadtxt("data/epipolar/house_points.txt")
points_left = [[x, y] for x,y in zip(correspondencePairs[:, 0], correspondencePairs[:, 1])]
points_right = [[x, y] for x,y in zip(correspondencePairs[:, 2], correspondencePairs[:, 3])]
F_houses = fundamental_matrix([points_left, points_right], showImages=True, pairOfImages=[house_image1, house_image2])
print(F_houses)


### 2c) calculate reprojection error
def calculateDistance(l, point):
    return abs(l[0]*point[0] + l[1]*point[1] + l[2]) / math.sqrt(l[0]**2 + l[1]**2)

def reprojection_error(F, matchedPoints):
    """ 
    Function calculates reprojection error of a 
    fundamental matrix F given two matching points.

    @param F fundamental matrix.
    @param matchedPoints in the form [[point1_x, point1_y], [point2_x, point2_y]]
    """
    point_left = np.append(np.array(matchedPoints[0]), 1)
    point_right = np.append(np.array(matchedPoints[1]), 1)
    l_left = np.dot(F, point_right)
    l_right = np.dot(F.T, point_left)

    dist1 = calculateDistance(l_left, point_left)
    dist2 = calculateDistance(l_right, point_right)

    return (dist1 + dist2) / 2


### 2c) calculate reprojection error - example 1.
print("Reprojection error is: " + str(reprojection_error(F_houses, [[67, 219], [85, 233]])) + " pixels.")



def computeReprojectionError(points, F, referencePoints):
    """ 
    Calculates reprojection error for given projected points, 
    fundamental matrix F and reference points.
    """
    reprojectionErrors = []
    for pt, ref_pt in zip(points, referencePoints):
        reprojectionErrors.append(reprojection_error(F, [ref_pt, pt]))

    return reprojectionErrors


### 2c) calculate reprojection error - example 1.
# reprojectionError_houses = 0

# for p_l, p_r in zip(points_left, points_right):
#     reprojectionError_houses += reprojection_error(F_houses, [p_r, p_l])
reprojectionError_houses = computeReprojectionError(points_left, F_houses, points_right)
reprojectionError = np.sum(np.array(reprojectionError_houses))
reprojectionError /= len(points_left)
print("Reprojection error is: " + str(reprojectionError)+ " pixels.")


desk_image1 = cv2.cvtColor(cv2.imread("data/desk/DSC02638.JPG"), cv2.COLOR_BGR2GRAY)
desk_image1 = desk_image1.astype(np.float64) * 1/255
desk_image2 = cv2.cvtColor(cv2.imread("data/desk/DSC02639.JPG"), cv2.COLOR_BGR2GRAY)
desk_image2 = desk_image2.astype(np.float64) * 1/255



### 2d) fundamental matrix estimation with RANSAC
def ransac_estimatingFundamentalMatrix(pts1, pts2, threshold=20, numberOfIterations=100):
    """ 
    Function implements RANSAC algorithm for robustly estimating a fundamental matrix.
    @param pts1  array of the coordinates of points of the first image, 
    elements are lists of the form [x_coordinate, y_coordinate].
    @param pts2  array of the coordinates of points of the second image, 
    elements are lists of the form [x_coordinate, y_coordinate].
    """
    numberOfMatches = len(pts1)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    F_result = []
    reprojectionErrors_result = -1

    for _ in range(numberOfIterations):
        # randomly select 8 matches 
        indices = random.sample(range(0, numberOfMatches), 8)

        # estimate the fundamental matrix
        pts1_chosen = pts1[indices, :]
        pts2_chosen = pts2[indices, :]
        F = fundamental_matrix([pts1_chosen, pts2_chosen])

        reprojectionErrors = np.array(computeReprojectionError(pts1, F, pts2))

        indices_inliers = reprojectionErrors < threshold
        current_inlierProbability = len(indices_inliers[indices_inliers == True])/len(pts1)
        pts1_inliers = pts1[indices_inliers, :]
        pts2_correspondencesToInliers = pts2[indices_inliers, :]

        if(len(indices_inliers[indices_inliers == True]) >= 8):
            F_inliers = fundamental_matrix([pts1_inliers, pts2_correspondencesToInliers])
            reprojectionError = np.sum(computeReprojectionError(pts1_inliers, F_inliers, pts2_correspondencesToInliers))

            if(reprojectionErrors_result == -1 or reprojectionError < reprojectionErrors_result):
                reprojectionErrors_result = reprojectionError
                F_result = F_inliers    
    return F_result


### 2d) fundamental matrix estimation with RANSAC - example part 1
[pts_desk1, pts_desk2] = modified_matching(desk_image1, desk_image2)
a4_utils.display_matches(desk_image1, pts_desk1, desk_image2, pts_desk2)


### 2d) fundamental matrix estimation with RANSAC - example part 2
F_desk = ransac_estimatingFundamentalMatrix(pts_desk1, pts_desk2)

drawEpipoles([desk_image1, desk_image2], [pts_desk1, pts_desk2], F_desk)


### ### ### EXERCISE 3 ### ### ###
### 3a) triangulation
def triangulate(correspondencePairs, calibrationMatrices):
    """  
    Function accepts a set of correspondence points and a pair of 
    calibration matrices as an input and returns the triangulated 3D points.
    """
    points_leftImage = np.array(correspondencePairs[0])
    x_l = points_leftImage[:, 0]
    y_l = points_leftImage[:, 1]
    
    points_rightImage = np.array(correspondencePairs[1])
    x_r = points_rightImage[:, 0] 
    y_r = points_rightImage[:, 1] 

    numberOfPoints = len(x_l)

    calibrationMatrix_left = calibrationMatrices[0]
    calibrationMatrix_right = calibrationMatrices[1]

    points_3d = []
    for i in range(numberOfPoints):
        lx = x_l[i]
        ly = y_l[i]
        lz = 1
        rx = x_r[i]
        ry = y_r[i]
        rz = 1
        
        A = []
        prod1 = np.dot([[0, -lz, ly], [lz, 0, -lx], [-ly, lx, 0]], calibrationMatrix_left)
        A.extend(prod1[0:2, :])
        prod2 = np.dot([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]], calibrationMatrix_right)
        A.extend(prod2[0:2, :])

        A = np.array(A)
        _, _, VT = np.linalg.svd(A)
        V = VT.T

        c = V[:, -1]
        X = V[:, -1] / V[-1][-1]
        X /= X[-1]
        X = X[0:3]

        points_3d.append(X)

    return points_3d


### 3a) triangulation - example.
calibrationMatrix_left = np.loadtxt("data/epipolar/house1_camera.txt")
calibrationMatrix_right = np.loadtxt("data/epipolar/house2_camera.txt")

points_3D = triangulate([points_left, points_right], [calibrationMatrix_left, calibrationMatrix_right])

T = np.array([[-1,0,0],[0,0,1],[0,-1,0]])

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')

points_3D = np.dot(points_3D,T)
for index, point in enumerate(points_3D):
    plt.plot([point[0]], [point[1]], [point[2]],'r.') 
    axes.text(point[0], point[1], point[2], str(index)) # plot indices
plt.show()