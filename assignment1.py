from UZ_utils import *

### ### ### ### ### ### MY UTILITY FUNCTIONS ### ### ### ### ### ### 

def displayImageAndBinaryMask(image, binaryMask, superTitle):
    """
    Function displays both grayscale image and its binary mask on one plot.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Grayscale image")

    plt.subplot(1, 2, 2)
    plt.imshow(binaryMask, cmap='gray')
    plt.title("Binary mask")

    plt.suptitle(superTitle)
    plt.show()

def displayTwoGSImages(image1, image2, title1, title2, superTitle):
    """
    Function displays two grayscale images on one plot.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2)

    plt.suptitle(superTitle)
    plt.show()

def displayThreeImages(image1, image2, image3, title1, title2, title3, suptitle):
    """
    Function displays three images as subplot in one row.
    """
    plt.subplot(1, 3, 1)
    plt.imshow(image1, cmap='gray')
    plt.title(title1, fontsize=10)

    plt.subplot(1, 3, 2)
    plt.imshow(image2, cmap='gray')
    plt.title(title2, fontsize=10)

    plt.subplot(1, 3, 3)
    plt.imshow(image3, cmap='gray')
    plt.title(title3, fontsize=10)

    plt.suptitle(suptitle)
    plt.show()  

### ### ### ### ### ### EXERCISE1 ### ### ### ### ### ### 

### 1a)
# reading image and displaying it
I = imread('images/umbrellas.jpg')
imshow(I)

### 1b)
# extracting image channels
redChannel = I[:, :, 0]
greenChannel = I[:, :, 1]
blueChannel = I[:, :, 2]

# converting image to grayscale image
grayScaleImage = (redChannel + greenChannel + blueChannel)/3
plt.imshow(grayScaleImage, cmap='gray')
plt.title("Converting umbrellas.jpg to grayscale image")
plt.show()

### 1c)
# defining bounds of the cutout
heightLowerBound = 130
heightUpperBound = 260
widthLowerBound = 240
widthUpperBound = 450

# cutting and displaying a part of the loaded image
cutout=I[heightLowerBound:heightUpperBound, widthLowerBound:widthUpperBound, 1]
plt.imshow(cutout, cmap='gray')
plt.title("Cutout part of the image")
plt.show()

"""
Question: Why would you use different color maps?
Answer: We would use different color maps, based on what we want to visualize and emphasize in the image. 
    For example, different color maps use different light settings or they can differently 
    tackle the problems such as color banding, false details and the color blindness ambiguity.
"""

### 1d)
# inverting a cutout part of the image
cutoutHeight, cutoutWidth = cutout.shape
imageWithInvertedCutout = np.copy(I)
imageWithInvertedCutout[heightLowerBound:heightUpperBound, widthLowerBound:widthUpperBound, :] = 1 - I[heightLowerBound:heightUpperBound, widthLowerBound:widthUpperBound, :]

plt.imshow(imageWithInvertedCutout)
plt.title("Image with inverted cutout part")
plt.show()

"""
Question: How is inverting a grayscale value defined for uint8?
Answer: Inverting a grayscale value, defined for uint8, is done by substracting the value of each pixel from 255.
"""

### 1e)
# reading image in gray, range of values is [0, 255]
originalGSImage = np.asarray(Image.open("images/umbrellas.jpg").convert('L')) 

# converting the grayscale image to floating point type 
modifiedGSImage = originalGSImage.astype(np.float64)
# rescalling imaging values and convert it to uint8
modifiedGSImage = (modifiedGSImage / 255) * 63
modifiedGSImage = modifiedGSImage.astype(np.uint8)

# change between images isn't seen
plt.subplot(1, 2, 1)
plt.imshow(originalGSImage, cmap='gray')
plt.title("Original grayscale image")

plt.subplot(1, 2, 2)
plt.imshow(modifiedGSImage, cmap='gray')
plt.title("Modified grayscale image")
plt.suptitle("Change between images isn't visible")
plt.show()

# change between images is seen
plt.subplot(1, 2, 1)
plt.imshow(originalGSImage, vmax=255, cmap='gray')
plt.title("Original grayscale image")

plt.subplot(1, 2, 2)
plt.imshow(modifiedGSImage, vmax=255, cmap='gray')
plt.title("Modified grayscale image")
plt.suptitle("Change between images is visible")
plt.show()

### ### ### ### ### ### EXERCISE2 ### ### ### ### ### ###  

### 2a)
# loading image 
birdGSImage = np.asarray(Image.open("images/bird.jpg").convert('L')) 

# first implementation of getting binary mask
birdThreshold = 80
birdBinaryMask = np.copy(birdGSImage)
birdBinaryMask[birdBinaryMask < birdThreshold] = 0
birdBinaryMask[birdBinaryMask >= birdThreshold] = 1

# second implementation of getting binary mask
birdBinaryMask = np.copy(birdGSImage)
birdBinaryMask = np.where(birdBinaryMask < birdThreshold, 0, 1)

displayImageAndBinaryMask(birdGSImage, birdBinaryMask, "Creating binary mask with threshold = 80")

# experimenting with different threshold values
for threshold in [60, 70, 90]:
    birdBinaryMask = np.copy(birdGSImage)
    birdBinaryMask = np.where(birdBinaryMask < birdThreshold, 0, 1)

    displayImageAndBinaryMask(birdGSImage, birdBinaryMask, "Creating binary mask with threshold = " + str(threshold))

### 2b)
numberOfBins = 20
def myhist(gsImage, numberOfBins):
    """
    Function retruns a 1D array that represents the gsImage
    histogram, which size is equal to the number of bins.    
    """
    H = np.zeros(numberOfBins)
    imageAs1DVector = gsImage.reshape(-1)
    scalledIntensities = ((imageAs1DVector / 256) * numberOfBins).astype(np.uint8)

    for i in scalledIntensities:
        H[i] += 1
    return H

def myhistNormalized(gsImage, numberOfBins):
    """
    Function normalizes histogram got from function myhist.
    """
    hist = myhist(gsImage, numberOfBins).astype(np.float64)
    sum = np.sum(hist)
    normalizedHist = hist / np.sum(hist)
    return normalizedHist

# calculating and displaying histogram for number of bins equal to 20
plt.subplot(2, 2, 1)
plt.bar(np.arange(numberOfBins), myhist(birdGSImage, numberOfBins))
plt.title("Histogram")

# calculating and displaying normalized histogram for number of bins equal to 20
plt.subplot(2, 2, 2)
plt.bar(np.arange(numberOfBins), myhistNormalized(birdGSImage, numberOfBins))
plt.title("Normalized histogram")

"""
Question: The histograms are usually normalized by dividing the result by the sum of all cells. Why is that?
Answer: That way, value corresponding to the specific bin is equal 
        to the probability that any pixel of the image belongs to that bin. 
        As well, it is much easier to compare different images' histograms, when they are normalized.
"""

### 2c)
def myhistModified(gsImage, numberOfBins):
    """
    Function retruns a 1D array that represents the gsImage
    histogram, which size is equal to the number of bins.    
    """
    H = np.zeros(numberOfBins)
    imageAs1DVector = gsImage.reshape(-1).astype(np.float64)
    minValue = np.min(imageAs1DVector)
    maxValue = np.max(imageAs1DVector)
    scalledIntensities = ((imageAs1DVector / (maxValue - minValue + 1)) * numberOfBins).astype(np.uint8)

    for i in scalledIntensities:
        H[i] += 1
    return H

def myhistModifiedNormalized(gsImage, numberOfBins):
    """
    Function normalizes histogram got from function myhistModified.  
    """
    hist = myhistModified(gsImage, numberOfBins).astype(np.float64)
    sum = np.sum(hist)
    normalizedHist = hist / np.sum(hist)
    return normalizedHist

# calculating and displaying histogram got with modified function for number of bins equal to 20
plt.subplot(2, 2, 3)
plt.bar(np.arange(numberOfBins), myhistModified(birdGSImage, numberOfBins))
plt.title("Histogram, got with modified function")

# calculating and displaying normalized histogram got with modified function for number of bins equal to 20
plt.subplot(2, 2, 4)
plt.bar(np.arange(numberOfBins), myhistModifiedNormalized(birdGSImage, numberOfBins))
plt.title("Normalized histogram, got with modified function")
plt.show()

### 2d)
# loading images of the same scene in different lighting conditions
lowLightImage = np.asarray(Image.open("images/low_light.jpg").convert('L')) 
mediumLightImage = np.asarray(Image.open("images/medium_light.jpg").convert('L')) 
highLightImage = np.asarray(Image.open("images/high_light.jpg").convert('L')) 

displayThreeImages(lowLightImage, mediumLightImage, highLightImage, "Low light image", "Medium light image", "High light image", "")

# visualizing the histograms for all images
for numberOfBins in [20, 40]:
    plt.subplot(1, 3, 1)
    plt.bar(np.arange(numberOfBins), myhistNormalized(lowLightImage, numberOfBins))
    plt.title("Histogram for low light image")

    plt.subplot(1, 3, 2)
    plt.bar(np.arange(numberOfBins), myhistNormalized(mediumLightImage, numberOfBins))
    plt.title("Histogram for medium light image")
    
    plt.subplot(1, 3, 3)
    plt.bar(np.arange(numberOfBins), myhistNormalized(highLightImage, numberOfBins))
    plt.title("Histogram for high light image")

    plt.suptitle("Number of bins is " + str(numberOfBins))
    plt.show()

"""
Interpretation of the results:
The image with the smallest amount of light has a peak at the beginning of the histogram, 
which is caused by the fact that most of the pixels in the image are quite dark, due to low brightness.

The image with medium brightness has the most even distribution on the histogram,
the diversity of intensities on the image is noticeable, there are no extremes on the histogram.

The histogram related to the image with the highest brightness has two peaks at the beginning, 
because the central object is the brightest, while the pixels in the background are quite dark.
About 20% of pixels of the image (pixels corresponding to the central object) are distributed 
in the upper half of the histogram, as the illumination is focused on the central object.
"""

### 2e)
# Implementation of Otsu’s method for automatic threshold calculation
def otsuMethod(gsImage, numberOfBins=256):
    """
    Function implements Otsu’s method for automatic threshold calculation. 
    It accepts a grayscale image and returns the optimal threshold.
    """
    histogram = np.histogram(gsImage, np.arange(numberOfBins))[0]
    histogram = histogram.astype(np.float64)
    histogram /= np.sum(histogram)
    optimalThreshold = 0
    maxBetweenClassVariance = 0

    # inital probabilities of both classes
    w0 = 0
    w1 = 1
    # inital mean values of both classes
    mi0 = 0
    mi1 = np.sum([i*histogram[i] for i in range(numberOfBins - 1)]) / w1
    for i in range(numberOfBins - 1):
        mi0 *= w0
        mi1 *= w1

        # computing new values of w0, w1, mi0 and mi1
        w0 += histogram[i]
        w1 -= histogram[i]
        mi0 += i*histogram[i]
        mi1 -= i*histogram[i]
        mi0 /= w0
        if(w1 != 0):
            mi1 /= w1

        currentBetweenClassVariance = w0*w1*(mi0 - mi1)**2            
        if(currentBetweenClassVariance > maxBetweenClassVariance):
            optimalThreshold = i + 1
            maxBetweenClassVariance = currentBetweenClassVariance
        
    return optimalThreshold

umbrellasGSImage = np.asarray(Image.open("images/umbrellas.jpg").convert('L')) 
coinsGSImage = np.asarray(Image.open("images/coins.jpg").convert('L')) 
eagleGSImage = np.asarray(Image.open("images/eagle.jpg").convert('L'))

# computing optimal threshold values for images using Otsu's method
for image in [birdGSImage, umbrellasGSImage, coinsGSImage, eagleGSImage]:
    optimalThreshold = otsuMethod(image)

    binaryMask = np.copy(image)
    binaryMask = np.where(binaryMask < optimalThreshold, 0, 1)

    displayImageAndBinaryMask(image, binaryMask, "Creating binary mask with threshold = " 
    + str(optimalThreshold) + " computed with Otsu's method")

### ### ### ### ### ### EXERCISE3 ### ### ### ### ### ### 

### 3a)
# loading image mask.png
maskImage = np.asarray(Image.open("images/mask.png").convert('L')) 

for n in [3, 5]:
    # create a square n x n structuring element
    SE = np.ones((n,n), np.uint8) 
    # applying erosion to the image
    maskImageEroded = cv2.erode(maskImage, SE) 
    # applying dilation to the image
    maskImageDilated = cv2.dilate(maskImage, SE)
    displayThreeImages(maskImage, maskImageEroded, maskImageDilated, "Original image", "Applying erosion to mask.png", 
    "Applying dilation to mask.png", "Size of structuring element is " + str(n) + " x " + str(n))    
    
    # applying opening to the image
    maskImageOpenning = cv2.dilate(cv2.erode(maskImage, SE), SE)
    # applying closing to the image
    maskImageClosing = cv2.erode(cv2.dilate(maskImage, SE), SE)
    displayThreeImages(maskImage, maskImageOpenning, maskImageClosing, "Original image", "Applying openning to mask.png", 
    "Applying closing to mask.png", "Size of structuring element is " + str(n) + " x " + str(n))    

"""
Question: Based on the results, which order of erosion and dilation operations produces opening and which closing?
Answer: Opening is produced by applying erosion and then dilation. 
        Its effects are that it removes small objects, preserves rough spaces, 
        and it can also be used for filtering out structures, by choosing appropriate structuring element.
        
        Closing is produced by applying dilation and then erosion. 
        Its effects are that it fills holes (in thresholded image), and preserves original shapes.
"""
### 3b)
# loading image bird.jpg and computing binary mask
birdGSImage = np.asarray(Image.open("images/bird.jpg").convert('L')) 
binaryMask = np.where(birdGSImage < otsuMethod(birdGSImage), 0, 1).astype(np.uint8)

# cleaning binary mask using morphological operations
for n in [21, 23, 25]:
    # creating different types of structuring element
    SE1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (n, n))
    SE2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n, n))

    cleanedBirdImage1 = cv2.erode(cv2.dilate(binaryMask, SE1), SE1)
    cleanedBirdImage2 = cv2.erode(cv2.dilate(binaryMask, SE2), SE2)

    displayThreeImages(binaryMask, cleanedBirdImage1, cleanedBirdImage2, "Original image", 
    "Applying closing with SE def. by cv2.MORPH_CROSS", "Applying closing with SE def. by CV2.MORPH_ELLIPSE", 
    "Size of structuring element is " + str(n) + " x " + str(n))

### 3c)
birdImage = np.asarray(Image.open("images/bird.jpg").convert('RGB')) 
def immask(rgbImage, binaryMask):
    image = np.copy(rgbImage)
    image[binaryMask == 0] = 0
    return image

image = immask(birdImage, binaryMask)
plt.imshow(image)
plt.title("Image got with function immask from bird.jpg")
plt.show()

### 3d)
# loading image eagle.jpg and creating its binary mask
eagleImage = np.asarray(Image.open("images/eagle.jpg").convert('RGB'))
eagleGSImage = np.asarray(Image.open("images/eagle.jpg").convert('L'))
eagleBinaryMask = np.where(eagleGSImage < otsuMethod(eagleGSImage), 0, 1).astype(np.uint8)

displayThreeImages(eagleGSImage, eagleBinaryMask, immask(eagleImage, eagleBinaryMask), 
"Original RGB image", "Binary mask with threshold got with Otsu's method", 
"Image got with function immask", "")

"""
Question: Why is the background included in the mask and not the object? How would you fix that in general? 
(just inverting the mask if necessary doesn’t count)
Answer: Background is included in the mask, because the object is darker than the background, 
so automatically on binary mask, background will be colored white, while object will be colored black.
Fixes in general, besides inverting the mask if necessary, depend on the image, but can be 
to check which color is dominant in the image, what color is on the edges and what color is in the central part of the image.
"""

### 3e)
# loading image coins.jpg and creating its binary mask
coinsRGBImage = np.asarray(Image.open("images/coins.jpg").convert("RGB"))
coinsResult = np.asarray(Image.open("images/coins.jpg").convert("L"))

# creating image's binary mask and inverting it
coinsResult = np.where(coinsGSImage < 220, 1, 0).astype(np.uint8)

# removing objects with area greater than 700, using connected components
connectivity = 4
output = cv2.connectedComponentsWithStats(coinsResult, connectivity, cv2.CV_32S)
labels = output[1]
areas = output[2][1:, -1]
for i in range(len(areas)):
    if(areas[i] > 700):
        coinsResult[labels == (i + 1)] = 0

displayTwoGSImages(coinsRGBImage, coinsResult, "Original image", "Final result", "")
