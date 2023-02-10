#!/usr/bin/env python3

"""
RBE/CS549 Spring 202: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import glob
import cv2
import math
import matplotlib.pyplot as plt
import imutils
import sklearn.cluster
import numpy as np
from scipy.stats import multivariate_normal
import scipy.ndimage as ndi
import scipy
from sklearn.cluster import KMeans

np.set_printoptions(suppress=True, precision=3)

# img1 = cv2.imread('../BSDS500/Images/1.jpg')

"""
Creates a filter bank of Difference of Gaussian (DOG) filters for a given set of scales and orientations
scales: a list of scales for which the DOG filters are to be created
size: the size of the gaussian kernel
orientation: the number of orientations for which the DOG filters are to be created
returns: a list of DOG filters
"""


def create_DOG_filter_bank(scales, size, orientation):
    filterBank = []

    x = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y = np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    for scale in scales:
        gaussian = create_gaussian_kernel(scale, size)

        X = cv2.filter2D(gaussian, -1, x)
        Y = cv2.filter2D(gaussian, -1, y)

        for eachOrient in range(orientation):
            gaussian_current = (X * np.cos((eachOrient * 2 * np.pi / orientation)) + Y * np.cos(
                (eachOrient * 2 * np.pi / orientation)))
            filterBank.append(gaussian_current)

    return filterBank


""""
Creates a 2D Gaussian kernel with a given sigma and kernel size
sigma: the sigma value for the Gaussian kernel
kernel_size: the size of the kernel
"""


def create_gaussian_kernel(sigma, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd.")

    kernel = np.zeros((kernel_size, kernel_size))

    center = kernel_size // 2

    for x in range(kernel_size):
        for y in range(kernel_size):
            x_val = x - center
            y_val = y - center
            kernel[x, y] = np.exp(-(x_val ** 2 + y_val ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)


"""
Creates a filter bank of Difference of Gaussian (DOG) filters for a given set of scales and orientations
scales: a list of scales for which the DOG filters are to be created
size: the size of the gaussian kernel
order: the order of the derivative to be taken
"""


def oneD_gauss(sigma, mean, x, order):
    mean_x = np.array(x) - mean
    var = sigma ** 2

    gaussian = (1 / np.sqrt(2 * np.pi * var)) * (np.exp((- 1 * mean_x * mean_x) / (2 * var)))

    if order == 0:
        g_1d = gaussian
        return g_1d
    elif order == 1:
        g_1d = - gaussian * ((mean_x) / (var))
        return g_1d
    else:
        g_1d = gaussian * (((mean_x * mean_x) - var) / (var ** 2))
        return g_1d


"""
Creates a 2D Gaussian function with a given support and scales. 
scales: the standard deviation of the Gaussian function
sup: the support of the Gaussian function
"""


def twoD_gauss(sup, scales):
    var = scales * scales
    shape = (sup, sup)
    n, m = [(i - 1) / 2 for i in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
    return g


"""
Creates a 2D Laplacian of Gaussian (LoG) function with a given support and scales.
sup: support of the LoG function
scales: the standard deviation of the Gaussian function
"""


def twoD_lapaccian(sup, scales):
    var = scales * scales
    shape = (sup, sup)
    n, m = [(i - 1) / 2 for i in shape]
    x, y = np.ogrid[-m:m + 1, -n:n + 1]
    g = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x * x + y * y) / (2 * var))
    h = g * ((x * x + y * y) - var) / (var ** 2)
    return h


"""
Generates a 2D gaussian filter with the given scale, phasex and phasey values, based on the given points and support.
:param scale: scale value for the gaussian filter.
:param phasex: phase value for the x-axis of the gaussian filter.
:param phasey: phase value for the y-axis of the gaussian filter.
:param pts: array of points used to create the gaussian filter.
:param sup: support value for the gaussian filter.
:return: 2D gaussian filter of shape (sup, sup).
 """


def makefilter(scale, phasex, phasey, pts, sup):
    gx = oneD_gauss(3 * scale, 0, pts[0, ...], phasex)
    gy = oneD_gauss(scale, 0, pts[1, ...], phasey)
    image = gx * gy
    image = np.reshape(image, (sup, sup))
    return image


"""
Generates a set of filters for use in the Laplacian of Gaussian (LoG) filter bank.
:return: 3D array of filters of shape (sup, sup, nf) where nf is the number of filters in the filter bank.
"""


def makeLMfilters():
    sup = 49
    scalex = np.sqrt(2) * np.array([1, 2, 3])
    norient = 6
    nrotinv = 12

    nbar = len(scalex) * norient
    nedge = len(scalex) * norient
    nf = nbar + nedge + nrotinv
    F = np.zeros([sup, sup, nf])
    hsup = (sup - 1) / 2

    x = [np.arange(-hsup, hsup + 1)]
    y = [np.arange(-hsup, hsup + 1)]

    [x, y] = np.meshgrid(x, y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient) / norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c + 0, -s + 0], [s + 0, c + 0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts, orgpts)
            F[:, :, count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:, :, count + nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar + nedge
    scales = np.sqrt(2) * np.array([1, 2, 3, 4])

    for i in range(len(scales)):
        F[:, :, count] = twoD_gauss(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:, :, count] = twoD_lapaccian(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:, :, count] = twoD_lapaccian(sup, 3 * scales[i])
        count = count + 1

    return F


"""
Generates a Gabor kernel with specified parameters.
:param ksize: int, the size of the kernel
:param wvlength: int, wavelength of the sinusoidal function
:param theta: float, the orientation of the normal to the parallel stripes
:param offset: float, the phase offset
:param sigma: int, standard deviation of the Gaussian envelope
:param gamma: float, the spatial aspect ratio
:return: numpy array, the generated Gabor kernel
"""


def gaborKernel(ksize, wvlength=6, theta=0, offset=0, sigma=6, gamma=1):
    if (ksize % 2) != 1:
        return None

    sideWidth = int((ksize - 1) / 2)  # convert to int
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    a = -0.5 / (math.pow(sigma, 2))
    b = -(0.5 * math.pow(gamma, 2)) / (math.pow(sigma, 2))
    const = (2.0 * math.pi) / wvlength
    ct = np.cos(theta)
    st = np.sin(theta)

    for x in range(-sideWidth, sideWidth):  # use int values here
        for y in range(-sideWidth, sideWidth):  # use int values here
            X = x + sideWidth
            Y = y + sideWidth
            x_d = x * ct + y * st
            y_d = -x * st + y * ct
            kernel[X, Y] = np.exp(math.pow(x_d, 2) * a + math.pow(y_d, 2) * b) * np.cos(const * x_d + offset)

    return kernel


"""
Creates a Gabor filter bank with specified parameters.
:param maxSize: int, the size of the filter bank
:param scales: list of lists, the scales of the Gabor kernels
:param orientation: int, the number of orientations of the Gabor kernels
:return: numpy array, the generated Gabor filter bank
"""


def gaborFilterBank(maxSize=37, scales=[[4, 4], [6, 4], [8, 6], [10, 8], [12, 14]], orientation=8):
    index = 0
    gaborfilterBank = np.zeros((maxSize, int(maxSize), len(scales) * orientation), dtype=np.float32)
    rotateAngle = 3.14159 / orientation
    for scale in scales:
        wv = scale[0]
        sig = scale[1]
        for i in range(orientation):
            gaborfilterBank[:, :, index] = gaborKernel(int(maxSize), wvlength=int(wv), theta=i * rotateAngle,
                                                       sigma=int(sig))
            index += 1
    return gaborfilterBank


"""
Creates a texton map of an input image using specified filters.
:param k: int, the number of clusters in the k-means algorithm
:param image: numpy array, the input image
:param dog: numpy array, the Difference of Gaussian filter bank
:param makeLMfilters: numpy array, the Laplacian of Gaussian filter bank
:param gabor: numpy array, the Gabor filter bank
:return: numpy array, the generated texton map
"""


def textonMap(k, image, dog, makeLMfilters, gabor):
    r, g, b = image.shape
    image_textron = np.array(image)
    filters = [dog, makeLMfilters, gabor]
    for bank in filters:
        for eachFilter in range(len(bank)):
            filtered = cv2.filter2D(image, -1, bank[eachFilter])
            image_textron = np.dstack((image_textron, filtered))

    final_texton = image_textron[:, :, 3:]
    x, y, z = final_texton.shape
    final_texton = final_texton.reshape((r * g), z)

    kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=2)
    kmeans.fit(final_texton)
    labels = kmeans.predict(final_texton)
    labels = labels.reshape([x, y])
    plt.imshow(labels)

    return labels


"""
Creates a brightness map of an input image using k-means clustering.
:param image: numpy array, the input image
:param clusters: int, the number of clusters in the k-means algorithm
:return: numpy array, the generated brightness map"""


def getBrightnessMap(image, clusters=16):
    reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(reshaped)
    label = kmeans.labels_
    label = label.reshape(image.shape[:2])
    return label


"""
Creates a color map of an input image using k-means clustering.
:param image: numpy array, the input image
:param clusters: int, the number of clusters in the k-means algorithm
:return: numpy array, the generated color map"""


def colorMap(image, clusters=16):
    reshaped = image.reshape((-1, 3))

    # Perform k-means clustering using sklearn
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(reshaped)
    label = kmeans.labels_
    label = label.reshape(image.shape[0], image.shape[1])
    plt.imshow(label)
    return label


"""
Generates a half disk mask of given radius and orientation.
:param radius: radius of the half disk mask
:param orient: orientation of the half disk mask in degrees
:return hd: half disk mask of given radius and orientation 
"""


def hdGen(radius, orient):
    size = 2 * radius + 1
    hd = np.zeros([size, size])
    for i in range(radius):
        for j in range(size):
            r = np.square(i - radius) + np.square(j - radius)
            if r <= np.square(radius):
                hd[i, j] = 1
    hd = imutils.rotate(hd, orient)
    hd[hd <= 0.5] = 0
    hd[hd > 0.5] = 1
    plt.imshow(hd)
    return hd


"""
Generates a list of half disk masks of different orientations.
:param radius : list of radii of the half disk masks
:param hdmOrientations: number of orientations of the half disk masks
:return hdMasks: list of half disk masks of different orientations
"""


def halfDiskMasks(radius, hdmOrientations):
    hdMasks = []
    for radii in radius:
        mask = np.zeros((radii * 2 + 1, radii * 2 + 1), dtype=np.float32)

        for i in range(radii):
            x = math.pow((i - radii), 2)
            for j in range(radii * 2 + 1):
                if x + math.pow((j - radii), 2) < math.pow(radii, 2):
                    mask[i, j] = 1

        rotateAngle = 360.0 / hdmOrientations
        for i in range(hdmOrientations):
            rotated = ndi.interpolation.rotate(mask, -i * rotateAngle, reshape=False)
            rotated[rotated > 1] = 1
            rotated[rotated < 0] = 0
            ret, rotated = cv2.threshold(rotated, 0.5, 1, cv2.THRESH_BINARY)

            # Rotated pair
            rotated_p = ndi.interpolation.rotate(mask, -i * rotateAngle - 180, reshape=False)
            rotated_p[rotated_p > 1] = 1
            rotated_p[rotated_p < 0] = 0
            ret, rotated_p = cv2.threshold(rotated_p, 0.5, 1, cv2.THRESH_BINARY)

            hdMasks.append(rotated)
            hdMasks.append(rotated_p)

    return hdMasks


"""
Computes Chi-squared gradient of an image using a given bank of half disk masks.
:param image: input image
:param chi_bins: number of bins to use for Chi-squared gradient computation
:param hdBank: list of half disk masks to use for Chi-squared gradient computation
:returns t_grad: Chi-squared gradient of the input image
        
"""


def chi2Gradient(image, chi_bins, hdBank):
    copy = image
    g = []
    h = []
    bank_length = len(hdBank) / 2
    for bl in range(int(bank_length)):
        chi_sqr_dist = image * 0
        mask_1 = hdBank[2 * bl]
        mask_2 = hdBank[2 * bl + 1]
        for eachBin in range(chi_bins):
            mask_image = np.ma.MaskedArray(image, image == eachBin)
            mask_image = mask_image.mask.astype(np.int64)
            g = cv2.filter2D(mask_image, -1, mask_1)
            h = cv2.filter2D(mask_image, -1, mask_2)
            chi_sqr_dist = chi_sqr_dist + ((g - h) ** 2 / (g + h + np.exp(-7)))

        copy = np.dstack((copy, chi_sqr_dist / 2))
    t_grad = np.mean(copy, axis=2)

    return t_grad


"""
Following 5 functions are used for visualization and saving resultss.
"""


def save_DoG(dog):
    plt.subplots(int(len(dog) / 5), 5, figsize=(15, 15))
    for d in range(len(dog)):
        plt.subplot(len(dog) / 5, 5, d + 1)
        plt.axis('off')
        plt.imshow(dog[d], cmap='gray')
    plt.savefig('../Results/DoG.png')
    plt.close()


def save_LM(makeLMfilters):
    x, y, r = makeLMfilters.shape
    plt.subplots(4, 12, figsize=(20, 20))
    for l in range(r):
        plt.subplot(4, 12, l + 1)
        plt.axis('off')
        plt.imshow(makeLMfilters[:, :, l], cmap='binary')
    plt.savefig('../Results/LM.png')
    plt.close()


def save_Gabor(gabor):
    fig = plt.figure()
    for i in range(1, 41):
        ax = fig.add_subplot(5, 8, i)
        plt.imshow(gabor[:, :, i - 1], interpolation='none', cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Gabor Filter Bank", fontsize=20)
    plt.savefig('../Results/Gabor.png')
    plt.close()


def save_HalfDisk(halfDisk):
    fig2 = plt.figure()
    for i in range(1, len(halfDisk) + 1):
        ax = fig2.add_subplot(6, 8, i)
        plt.imshow(halfDisk[i - 1], interpolation='none', cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle("Half Disk Filter Bank", fontsize=20)
    plt.savefig('../Results/HDisk.png')
    plt.close()


def save_brighnessMap(bMap):
    fig4 = plt.figure()
    fig4.suptitle('Brightness Map', fontsize=20)
    plt.imshow(bMap)
    plt.savefig('../Results/brightness_map_1.png')
    plt.close()


def getImages():
    image_list = []
    for i in range(10):
        for file in glob.glob('../BSDS500/Images/' + str(i + 1) + '.jpg'):
            im = cv2.imread(file)
            if im is not None:
                image_list.append(im)
            else:
                print("Couldn't load the image :( ", file)
    return image_list
all_images = getImages()

def main():


    """ 
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    dog_filter_bank = create_DOG_filter_bank([7, 4, 10], 49, 15)
    print("\nGenerating DOG Bank...\n")
    save_DoG(dog_filter_bank)

    """
    Generate Leung-Malik Filter Bank: (makeLMfilters)
    Display all the filters in this filter bank and save image as makeLMfilters.png,
    use command "cv2.imwrite(...)"
    """
    lm_filter_bank = makeLMfilters()
    print("\nGenerating makeLMfilters Bank...\n")
    save_LM(lm_filter_bank)

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    gabor_filter_bank = gaborFilterBank()
    print("\nGenerating GABOR Bank...\n")
    save_Gabor(gabor_filter_bank)

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    hd_filter_bank = halfDiskMasks([5, 10, 15], 8)
    print("\nGenerating Half Disk Masks...\n")
    save_HalfDisk(hd_filter_bank)

    """ 
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
    for i, image in enumerate(all_images):
        print("\nGenerating Texton Maps...\n")
        texton_map = textonMap(64, all_images[i], dog_filter_bank, lm_filter_bank, gabor_filter_bank)
        plt.imsave('../Results/Maps/TextonMap_' + str(i + 1) + '.png', texton_map)

        """
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)
		"""
        print("\nGenerating Texton Gradients...\n")
        texton_gradient = chi2Gradient(texton_map, 64, hd_filter_bank)
        fig6 = plt.figure()
        # plt.imshow(texton_gradient, cmap='gray')
        # fig6.suptitle("Texton Gradient", fontsize=20)
        plt.imsave('../Results/Gradients/Tg_e_' + str(i + 1) + '.png', texton_gradient)
        """
		Generate Brightness Map
		Perform brightness binning 
		"""
        brightness_map = getBrightnessMap(all_images[i])
        plt.imsave('../Results/Maps/brightness_map_' + str(i + 1) + '.png', brightness_map, cmap='binary')

        """
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
        brightness_gradient = chi2Gradient(brightness_map, 16, hd_filter_bank)
        plt.imsave('../Results/Gradients/Bg_' + str(i + 1) + '.png', brightness_gradient, cmap='binary')

        """
		Generate Color Map
		Perform color binning or clustering
		"""
        color_map = colorMap(all_images[i])
        plt.imsave('../Results/Maps/color_map_' + str(i + 1) +'.png', color_map)

        """
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
        color_gradient = chi2Gradient(color_map, 16, hd_filter_bank)
        plt.imsave('../Results/Gradients/Cg_' + str(i + 1) +'.png', color_gradient)

        """
		Read Sobel Baseline
		use command "cv2.imread(...)"
		"""
        sobel_baseLine = cv2.imread('../BSDS500/SobelBaseline/' + str(i + 1) +'.png', 0)
        """
		Read Canny Baseline
		use command "cv2.imread(...)"
		"""
        canny_baseLine = cv2.imread('../BSDS500/CannyBaseline/' + str(i + 1) +'.png', 0)
        """
		Combine responses to get pb_lite-lite output
		Display pb_liteLite and save image as pb_liteLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
        pb_lite = (texton_gradient + brightness_gradient + color_gradient) / 3

        pb_lite_out = np.multiply(pb_lite, (0.5 * sobel_baseLine + 0.5 * canny_baseLine))
        print("\nSaving pb_lite-Lite Output ...\n")
        plt.imshow(pb_lite_out, cmap='gray')
        plt.imsave('../Results/PB-Lite/pb_liteLite_' + str(i + 1) +'.png', pb_lite_out)

if __name__ == '__main__':
    main()
