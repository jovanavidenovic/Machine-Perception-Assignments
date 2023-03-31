
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import a4_utils
import random

### ### ### MY UTILITY FUNCTIONS ### ### ###
def displayMultipleImages(arrayOfImages, colorMap="", suptitle=""):
    """
    Function displays multiple images on one plot.
    Images are given in array, each element in arrayOfImages is 2 element array, 
    representing image and corresponding title. 
    @param colorMap defines the color map used in plt.imshow. 
        If it is "", default color map is applied.
    """
    numberOfColons = int(np.ceil(len(arrayOfImages) / 2))
    i = 1
    for element in arrayOfImages:
        plt.subplot(2, numberOfColons, i)
        if(colorMap != ""):
            plt.imshow(element[0], cmap=colorMap)
        else:
            plt.imshow(element[0])
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




### ### ### MY UTILITY FUNCTIONS ### ### ###
def displayMultipleImages_markCorners(arrayOfImages, colorMap="", suptitle=""):
    """
    Function displays multiple images on one plot.
    Images are given in array, each element in arrayOfImages is 4 element array, 
    [gsImage, hessianDeterminant, nms_hessianDeterminant, title] 
    element hessianDeterminant should be used for plotting "x" on the gsImage.
    @param colorMap defines the color map used in plt.imshow. 
        If it is "", default color map is applied.

    Function displays image showing Hessian determinant and corresponding marked image
    in one column. Number of columns is equal to the length of @param arrayOfImages.
    """
    numberOfColons = len(arrayOfImages)
    i = 1
    for element in arrayOfImages:
        plt.subplot(2, numberOfColons, i)
        if(colorMap != ""):
            plt.imshow(element[1], cmap=colorMap)
        else:
            plt.imshow(element[1])
        plt.title(element[3])

        plt.subplot(2, numberOfColons, i + numberOfColons)
        coordinates = np.nonzero(element[2])
        # marking corners with "x" on the image
        if(colorMap != ""):
            plt.imshow(element[0], cmap=colorMap)
        else:
            plt.imshow(element[0])
            
        for corner_y, corner_x in zip(coordinates[0], coordinates[1]):
            plt.scatter(corner_x, corner_y, s=10, c='red', marker='x')

        i += 1
    
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()


### ### ### EXERCISE 1 - FEATURE POINTS DETECTION ### ### ###
### 1a) Hessian key point detector
def hessian_points(gsImage, threshold, sigma=1):
    """
    Computes a Hessian determinant for each pixel of the input gs image.
    
    Applies postprocessing - applies threshold and non maxima suppression 
    and marks found corners with red circles on rgb image.
    
    @param image input grayscale image
    @param sigma sigma used in Gaussian filtering.
    @return list of the form [gsImage, hessianDeterminant, hessianDeterminant_nms, title] 
    """
    [image_dxx, image_dxy, _, image_dyy] = partialDerivatives_secondOrder(gsImage, sigma)
    hessianDeterminant = np.array(image_dxx*image_dyy - image_dxy**2)

    # non-maxima suppression
    # only retain pixels reponses whose value is higher than threshold
    hessianDeterminant_thresholded = hessianDeterminant.copy()
    hessianDeterminant_thresholded[hessianDeterminant < threshold] = 0 
    # only retain pixels whose value is higher than any in its neighborhood
    hessianDeterminent_nms = nonmaxima_suppression_box(hessianDeterminant_thresholded)

    return [gsImage, hessianDeterminant, hessianDeterminent_nms, "sigma=" + str(sigma)]

gsImage_testPoints = cv2.cvtColor(cv2.imread("data/graf/graf_a.jpg"), cv2.COLOR_BGR2GRAY)
gsImage_testPoints = gsImage_testPoints.astype(np.float64)
gsImage_testPoints *= 1/255

image_testPoints_sigma3 = hessian_points(gsImage_testPoints, 0.004, 3)
image_testPoints_sigma6 = hessian_points(gsImage_testPoints, 0.004, 6)
image_testPoints_sigma9 = hessian_points(gsImage_testPoints, 0.004, 9)


displayMultipleImages_markCorners([image_testPoints_sigma3, image_testPoints_sigma6, image_testPoints_sigma9], 
colorMap="gray", suptitle="Hessian corner detector")


"""
Question: What kind of structures in the image are detected by the algorithm?
How does the parameter sigma affect the result?

Answer: Corners are detected by using this algorithm, larger sigma affects in smaller number of corners detected, as areas are more smoothed.
"""


### 1b) Harris feature point detector
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

# loading and normalizing image
gsImage_testPoints = cv2.cvtColor(cv2.imread("data/graf/graf_a.jpg"), cv2.COLOR_BGR2GRAY)
gsImage_testPoints = gsImage_testPoints.astype(np.float64)
gsImage_testPoints *= 1/255

image_testPoints_sigma3 = harris_detector(gsImage_testPoints, threshold=1e-6, sigma=3, sigma1=4.8)
image_testPoints_sigma6 = harris_detector(gsImage_testPoints, threshold=1e-6, sigma=6, sigma1=9.6)
image_testPoints_sigma9 = harris_detector(gsImage_testPoints, threshold=1e-6, sigma=9, sigma1=14.4)

displayMultipleImages_markCorners([image_testPoints_sigma3, image_testPoints_sigma6, image_testPoints_sigma9], 
colorMap="gray", suptitle="Harris corner detector")    


"""
Question: Do the feature points of both detectors appear on the same structures in the image?
Answer: No.
"""



### ### ### EXERCISE 2 - MATCHING LOCAL REGIONS ### ### ###
### 2a)
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


# loading and normalizing images
image_grafAsmall = cv2.cvtColor(cv2.imread("data/graf/graf_a_small.jpg"), cv2.COLOR_BGR2GRAY)
image_grafAsmall = image_grafAsmall.astype(np.float64)
image_grafAsmall *= 1/255

image_grafBsmall = cv2.cvtColor(cv2.imread("data/graf/graf_b_small.jpg"), cv2.COLOR_BGR2GRAY)
image_grafBsmall = image_grafBsmall.astype(np.float64)
image_grafBsmall *= 1/255

# finding coordinates of feature points using Harris feature point detector.
featurePointsDetector_grafASmall = harris_detector(image_grafAsmall, 1e-6, 6, 9.6)
fPCoords_grafASmall = np.nonzero(featurePointsDetector_grafASmall[2])

featurePointsDetector_grafBSmall = harris_detector(image_grafBsmall, 1e-6, 6, 9.6)
fPCoords_grafBSmall = np.nonzero(featurePointsDetector_grafBSmall[2])

# calculating descriptors
desc_grafAsmall = a4_utils.simple_descriptors(image_grafAsmall, fPCoords_grafASmall[0], fPCoords_grafASmall[1])
desc_grafBsmall = a4_utils.simple_descriptors(image_grafBsmall, fPCoords_grafBSmall[0], fPCoords_grafBSmall[1])

# finding correspondences
correspondences = find_correspondences(desc_grafAsmall, desc_grafBsmall)

# calculating coordinates of the correspondences
pts_grafASmall = [[fPCoords_grafASmall[1][index], fPCoords_grafASmall[0][index]] for index in correspondences[:, 0]]
pts_grafBSmall = [[fPCoords_grafBSmall[1][index], fPCoords_grafBSmall[0][index]] for index in correspondences[:, 1]]

a4_utils.display_matches(image_grafAsmall, pts_grafASmall, image_grafBsmall, pts_grafBSmall)


### 2b) simple feature point matching algorithm
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
    featurePointsDetector1 = harris_detector(image1, 1e-6, 3, 4.8)
    fPCoords1 = np.nonzero(featurePointsDetector1[2])

    featurePointsDetector2 = harris_detector(image2, 1e-6, 3, 4.8)
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


### 2c) simple method for eliminating incorrect matches
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
    featurePointsDetector1 = harris_detector(image1, 1e-6, 3, 4.8)
    fPCoords1 = np.nonzero(featurePointsDetector1[2])

    featurePointsDetector2 = harris_detector(image2, 1e-6, 3, 4.8)
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

    return [pts1, pts2]

### 2c) simple method for eliminating incorrect matches - example
[pts_grafASmall, pts_grafBSmall] = simple_matching(image_grafAsmall, image_grafBsmall)
[pts_modified_grafASmall, pts_modified_grafBSmall] = modified_matching(image_grafAsmall, image_grafBsmall)
a4_utils.display_matches(image_grafAsmall, pts_grafASmall, image_grafBsmall, pts_grafBSmall)
a4_utils.display_matches(image_grafAsmall, pts_modified_grafASmall, image_grafBsmall, pts_modified_grafBSmall)

### 2e) detecting key points in a video
def displayKeyPointsOnVideo(path):
    """ 
    @param path path of the video
    Shows video with detected key points.
    """
    cap = cv2.VideoCapture(path)
    det = cv2.FastFeatureDetector_create(threshold=50)        

    while(cap.isOpened()):
        ret, frame = cap.read()
        if (ret == True):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = det.detect(frame, None)
            frame = cv2.drawKeypoints(image=frame, keypoints=keypoints, outImage=None)
            cv2.imshow('frame', frame)

            if (cv2.waitKey(25) & 0xFF == ord('q')):
                break
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()


### 2e) detecting key points in a video
displayKeyPointsOnVideo("data/video.mp4")


### ### ### EXERCISE 3 - HOMOGRAPHY ESTIMATION ### ### ###
def estimate_homography(matchedFeaturePoints):
    """
    Function approximates a homography between two images using 
    a given set of matched feature points.

    @param matchedFeaturePoints given set of matched feature points,
    represented by a list of the form [pts1, pts2], where both elements 
    are lists of the form [x_coordinate, y_coordinate].
    """
    # extracting point coordinate from matchedFeaturePoints
    xy_r = np.array(matchedFeaturePoints[0])
    x_r = xy_r[:, 0]
    y_r = xy_r[:, 1]
    xy_t = np.array(matchedFeaturePoints[1])
    x_t = xy_t[:, 0]
    y_t = xy_t[:, 1]

    # calculating number of matches
    numberOfMatches = len(xy_r)
    
    # constructing matrix A for equation Ah=0
    A = np.zeros((2*numberOfMatches, 9))
    for i in range(0, 2*numberOfMatches, 2):
        points_i = int(i / 2)
        A[i, :] = np.array([x_r[points_i], y_r[points_i], 1, 0, 0, 0, -x_t[points_i]*x_r[points_i], -x_t[points_i]*y_r[points_i], -x_t[points_i]])
        A[i + 1, :] = np.array([0, 0, 0, x_r[points_i], y_r[points_i], 1, -y_t[points_i]*x_r[points_i], -y_t[points_i]*y_r[points_i], -y_t[points_i]])
    
    # performing a matrix decomposition using the SVD algorithm.
    _, _, VT = np.linalg.svd(A)
    V = VT.T

    # computing vector h and reshape it to 3x3 H matrix
    h = V[:, -1] / V[-1][-1]
    H = np.reshape(h, (3, 3))    
    return H


### 3a) homography estimation - example
# loading images
image_newYork_a = cv2.cvtColor(cv2.imread("data/newyork/newyork_a.jpg"), cv2.COLOR_BGR2GRAY)
image_newYork_b = cv2.cvtColor(cv2.imread("data/newyork/newyork_b.jpg"), cv2.COLOR_BGR2GRAY)

# loading correspondence pairs
correspondencePairs = np.loadtxt("data/newyork/newyork.txt")
points_a = [[x, y] for x,y in zip(correspondencePairs[:, 0], correspondencePairs[:, 1])]
points_b = [[x, y] for x,y in zip(correspondencePairs[:, 2], correspondencePairs[:, 3])]

# display matches
a4_utils.display_matches(image_newYork_a, points_a, image_newYork_b, points_b)

# calculating homography matrix
H = estimate_homography([points_a, points_b])
# print(H)

# transform the first image to the plane of the second image
transformedImage = cv2.warpPerspective(image_newYork_a, H, image_newYork_a.shape)
displayMultipleImages([[transformedImage, ""]], colorMap="gray", suptitle="Transforming image newyork_a\nto image newyork_b using homography")

### 3b) RANSAC algorithm for robustly estimating a homography
def computeReprojectionError(points, H, referencePoints):
    """ 
    @return numpy array of reprojection errors for all points that can be calculated by multiplying
    the point’s coordinates with the homography matrix and comparing the result
    to the reference point from the other image.
    Comparision is done using Euclidean distance.
    """
    errors = []
    for i in range(len(points)):
        point = points[i]
        # converting to homogenous coordinates
        point = np.append(point, [1])
        point = np.array([point])
        point = point.T       
        projectedPoint_homogenous = np.dot(H, point)
        # converting to cartesian coordinates
        projectedPoint_cartesian = (projectedPoint_homogenous/projectedPoint_homogenous[-1])[0:2]
        x1 = int(projectedPoint_cartesian[0])
        x2 = referencePoints[i][0]
        y1 = int(projectedPoint_cartesian[1])
        y2 = referencePoints[i][1]
        errors.append(math.sqrt((x1 - x2)** 2 + (y1 - y2)**2))
    errors = np.array(errors)
    return errors

def ransac_estimatingHomography(pts1, pts2, earlyTermination=False, numberOfIterations=100, threshold=2):
    """ 
    Function implements RANSAC algorithm for robustly estimating 
    a homography between matching points.
    @param pts1  array of the coordinates of points of the first image, 
    elements are lists of the form [x_coordinate, y_coordinate].
    @param pts2  array of the coordinates of points of the second image, 
    elements are lists of the form [x_coordinate, y_coordinate].
    """
    numberOfMatches = len(pts1)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    H_result = []
    reprojectionErrors_result = -1
    inlierProbability = 0

    for i in range(numberOfIterations):
        # randomly select 4 matches 
        indices = random.sample(range(0, numberOfMatches), 4)

        # estimate the homography matrix
        pts1_chosen = pts1[indices, :]
        pts2_chosen = pts2[indices, :]
        H = estimate_homography([pts1_chosen, pts2_chosen])

        # determine the inliers for the estimated homography
        reprojectionErrors = computeReprojectionError(pts1, H, pts2)

        indices_inliers = reprojectionErrors < threshold
        current_inlierProbability = len(indices_inliers[indices_inliers == True])/len(pts1)
        inlierProbability += current_inlierProbability
        pts1_inliers = pts1[indices_inliers, :]
        pts2_correspondencesToInliers = pts2[indices_inliers, :]

        if(current_inlierProbability >= 0.6):
            H_inliers = estimate_homography([pts1_inliers, pts2_correspondencesToInliers])
            reprojectionError = np.sum(computeReprojectionError(pts1_inliers, H_inliers, pts2_correspondencesToInliers))

            if(reprojectionErrors_result == -1 or reprojectionError < reprojectionErrors_result):
                reprojectionErrors_result = reprojectionError
                H_result = H_inliers
        
        if(earlyTermination and 0.85 <= len(indices_inliers[indices_inliers == True])/len(pts1)):
            print(i)
            return H_result

    if(not earlyTermination):
        inlierProbability /= numberOfIterations
        print(inlierProbability)
    else:
        print(numberOfIterations)
    return H_result


### 3b) RANSAC algorithm for robustly estimating a homography - example
# find a set of matched points for New York images
image_newYork_a_normalized = image_newYork_a.astype(np.float64) * 1/255
image_newYork_b_normalized = image_newYork_b.astype(np.float64) * 1/255
[points_a, points_b] = modified_matching(image_newYork_a_normalized, image_newYork_b_normalized)
a4_utils.display_matches(image_newYork_a_normalized, points_a, image_newYork_b_normalized, points_b)
H_ransac = ransac_estimatingHomography(points_a, points_b)

# transform the first image to the plane of the second image
transformedImage = cv2.warpPerspective(image_newYork_a_normalized, H_ransac, image_newYork_a_normalized.shape)
displayMultipleImages([[transformedImage, ""]], colorMap="gray", suptitle="Transforming image newyork_a\nto image newyork_b using RANSAC estimation of homography")


### 3c)
# estimated number of iterations is 12
# estimated inlier probability is around 0.75
# implementing early termination - when inlier ratio eaches expected ratio of inliers.
H_ransac_earlyTermination = ransac_estimatingHomography(points_a, points_b, earlyTermination=True)

# transform the first image to the plane of the second image
transformedImage = cv2.warpPerspective(image_newYork_a_normalized, H_ransac_earlyTermination, image_newYork_a_normalized.shape)
displayMultipleImages([[transformedImage, ""]], colorMap="gray", suptitle="Transforming image newyork_a to image newyork_b using RANSAC \n estimation of homography with enabled early termination")


### 3e) my own implementation for cv2.warpPerspective()
def warp_perspective(image, H):
    """ 
    @param image original image
    @param H homography matrix to remap the image

    @return image remapped by homography
    """
    (height, width) = image.shape
    projectedImage = np.zeros((height, width))

    H_inverse = np.linalg.inv(H)
    for i in range(height):
        for j in range(width):
            # projectedPoint_homogenous is of the form [x', y', 1]
            projectedPoint_homogenous = np.array([j, i, 1]) 
            # point_homogeonous is of the form [x, y, z] 
            point_homogenous = np.dot(H_inverse, projectedPoint_homogenous)
            point_cartesian = (point_homogenous/point_homogenous[-1])[0:2]
            point_x = int(math.ceil(point_cartesian[0]))
            point_y = int(math.ceil(point_cartesian[1]))
            if(point_x >= 0 and point_y >= 0 
            and point_x < width and point_y < width):
                projectedImage[i, j] = image[point_y, point_x]
    return projectedImage


### 3e) my own implementation for cv2.warpPerspective() - example
# transform the first image to the plane of the second image
transformedImage = warp_perspective(image_newYork_a_normalized, H_ransac)
displayMultipleImages([[transformedImage, ""]], colorMap="gray", suptitle="Transforming image newyork_a\nto image newyork_b using RANSAC estimation of homography.")


