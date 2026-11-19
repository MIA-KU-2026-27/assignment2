import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import diff_ops


def load_image(filename):
    image = plt.imread(filename)
    # input image is color png depicting grayscale, just use first plane from here on
    image = image[:, :, 1].astype(np.float64)
    # after reading and converting to float, image is between 0 and 1, we want to compute within the range 0 and 255
    image *= 255.0
    #print(image.shape)
    #print('image: min = ', np.min(image), ' max = ', np.max(image))
    return image




def rmse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return np.sqrt(mse)



def main():
    filename = 'CTThoraxSlice256.png'

    # load image
    image = load_image(filename)

    # Blurring of the ORIGINAL image. sigma of 3 for spatial smoothing, leads to strong blurring
    blurred = ndimage.gaussian_filter(image, sigma=3)
    # Example for rmse computation
    print('RMSE (blurred):', rmse(image, blurred))

    # compute x and y gradient images of ORIGINAL image using forward difference gradient operators from diff_ops.py
    grad_x_fwdiff = diff_ops.dx_forward(image)
    grad_y_fwdiff = diff_ops.dy_forward(image)

    
    # add noise with std. dev. 30 to input image (input image has range [0,255])
    img_noise = 30 * np.random.randn(image.shape[0], image.shape[1]) + image
    # clip noised image to [0,255], since noising process adds values outside of range
    img_noise[img_noise > 255] = 255
    img_noise[img_noise < 0] = 0

    # plot what we have
    fig, axarr = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)

    axarr[0, 0].imshow(image, cmap=plt.cm.gray)
    axarr[0, 0].set_title('Original Image')
    axarr[0, 1].imshow(blurred, cmap=plt.cm.gray)
    axarr[0, 1].set_title('Original Img Gaussian Blur')
    axarr[1, 0].imshow(grad_x_fwdiff, cmap=plt.cm.gray)
    axarr[1, 0].set_title('Original Img X Gradient')
    axarr[1, 1].imshow(grad_y_fwdiff, cmap=plt.cm.gray)
    axarr[1, 1].set_title('Original Img Y Gradient')

    plt.show()






if __name__ == "__main__":
    main()


