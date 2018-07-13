from matplotlib import pyplot as plt
import numpy as np
import cv2


def compare(imgA, imgB, threshold=0.85, disp=100, verbose=True, save=True):
    """Image comparisson function."""

    # Verify sizes:
    if imgA.shape[0] < disp:
        print("ERROR: Displacement greater than image height.")
        print("(You can try again by reducing it, maybe.).")

    # Create a copies from originals:
    img_rgb = imgA.copy()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = imgB.copy()
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Create figure:
    if verbose:
        plt.figure(figsize=(15, 5))

    # Set auxiliary variables:
    total = []
    times = imgA.shape[0] // disp + 1

    # Compare:
    for i in range(times):
        k = 0
        if i * disp + disp >= imgA.shape[0]:
            template_sub = template[i * disp:, :]
        else:
            template_sub = template[i * disp:i * disp + disp, :]

        w, h = template_sub.shape[::-1]
        res = cv2.matchTemplate(img_gray, template_sub, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            k += 1
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        if k:
            total.append(max_val)

        if len(total):
            mean = len(total) / times
            sim_coeff = np.mean(total)
        else:
            mean = 0
            sim_coeff = 0

        # Display plot:
        if verbose:
            plt.subplot(141), plt.imshow(imgA)
            plt.title('Image A'), plt.xticks([]), plt.yticks([])
            plt.subplot(142), plt.imshow(imgB, cmap='gray')
            plt.title('Image B'), plt.xticks([]), plt.yticks([])
            plt.subplot(143), plt.imshow(img_rgb, cmap='gray')
            plt.title('Swapping result'), plt.xticks([]), plt.yticks([])
            plt.subplot(144), plt.imshow(template_sub, cmap='gray')
            plt.title('Template image'), plt.xticks([]), plt.yticks([])
            plt.suptitle('Measurement by template matching')
            plt.pause(0.01)
            plt.show(block=False)

    if verbose:
        plt.show()
        print()
        print("Matching blocks:", len(total))
        print("Matching vector:", total)
        print("Mean per swapping:", mean)
        print("Similitude coeff. (mean of blocks):", sim_coeff)

    if save:
        fig = plt.figure(figsize=(15, 7))
        plt.subplot(131), plt.imshow(imgA)
        plt.title('Image A'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(imgB, cmap='gray')
        plt.title('Image B'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(img_rgb, cmap='gray')
        plt.title('Swapping result'), plt.xticks([]), plt.yticks([])
        title = "Measurement by template matching: \n"
        res = "$\mu$={:.4} $sim. coeff.$={:.4}".format(mean, sim_coeff)
        plt.suptitle(title + res)
        plt.savefig("assets/res.png")

    return mean, sim_coeff


if __name__ == '__main__':
    imgA = cv2.imread('assets/img1.png')
    imgB = cv2.imread('assets/img2.png')
    compare(imgA, imgB, threshold=0.85, disp=100, verbose=True)
