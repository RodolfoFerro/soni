# Importing useful stuff:
from skimage.measure import compare_ssim
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np


class soni():
    """
    Similitude on Images.

    Based on:
    https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    """

    def __init__(self):
        self.mse = None
        self.ssim = None
        return

    def resize_imgs(self, imageA, imageB):
        lowest_W = min(imageA.shape[0], imageB.shape[0])
        lowest_H = min(imageA.shape[1], imageB.shape[1])
        imageA_ed = resize(imageA, (lowest_W, lowest_H))
        imageB_ed = resize(imageB, (lowest_W, lowest_H))
        return imageA_ed, imageB_ed

    def mean_squared_error(self, imageA, imageB):
        """Function to compute the Mean Squared Error."""

        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        return err

    def compare_images(self, imageA, imageB):
        if imageA.shape != imageB.shape:
            imageA_e, imageB_e = self.resize_imgs(imageA, imageB)
        else:
            imageA_e, imageB_e = imageA, imageB

        # compute the mean squared error and structural similarity
        # index for the images
        self.mse = self.mean_squared_error(imageA_e, imageB_e)
        self.ssim = compare_ssim(imageA_e, imageB_e)

        # setup the figure
        fig = plt.figure("Image comparisson", figsize=(30, 20))
        plt.suptitle("MSE: %.2f, SSIM: %.2f" %
                     (self.mse, self.ssim), fontsize=60)

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA_e, cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB_e, cmap=plt.cm.gray)
        plt.axis("off")

        # show the images
        plt.savefig('assets/out.png')


if __name__ == '__main__':
    img1 = rgb2gray(plt.imread('assets/img1.png'))
    img2 = rgb2gray(plt.imread('assets/img2.png'))
    soni = soni()
    soni.compare_images(img1, img2)
