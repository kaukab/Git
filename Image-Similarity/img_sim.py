from scipy import ndimage
import numpy as np
from scipy import misc
import PIL
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from io import StringIO
import urllib



os.chdir('_______')



img = Image.open("pool1.png" )

img.size

data = np.asarray( img, dtype='uint8' )

data.shape

data[:,:,3].shape

PIL.Image.fromarray(data[:,:,3])

PIL.Image.fromarray(data[:,:,2])

PIL.Image.fromarray(data[:,:,1])





def get_thumbnail(image, size=(128,128), stretch_to_fit=False, greyscale=False):

    " get a smaller version of the image - makes comparison much faster/easier"

    if not stretch_to_fit:

        image.thumbnail(size, Image.ANTIALIAS)

    else:

        image = image.resize(size); # for faster computation

    if greyscale:

        image = image.convert("L")  # Convert it to grayscale.

    return image





img1 = get_thumbnail(Image.open('pool1.png'))

img2 = get_thumbnail(Image.open('pool2.png'))



np.asarray(img1,  dtype='uint8' ).shape

np.asarray(img2,  dtype='uint8' ).shape



def mse(imageA, imageB):

 # the 'Mean Squared Error' between the two images is the

 # sum of the squared difference between the two images;

 # NOTE: the two images must have the same dimension

 imageA_np = np.asarray(imageA,  dtype='uint8' )[:,:,0:3]

 imageB_np = np.asarray(imageB,  dtype='uint8' )[:,:,0:3]

 err = np.sum((imageA_np.astype("float") - imageB_np.astype("float")) ** 2)

 err /= float(imageA_np.shape[0] * imageA_np.shape[1])

 # return the MSE, the lower the error, the more "similar"

 # the two images are

 return err





mse(img1, img2)



# normalized cross correlation

temp = []



def get_pixel_avg_vec(imageobj):

    from numpy import average

    vector = []

    for pixel_tuple in imageobj.getdata():

        vector.append(average(pixel_tuple))

    return vector



get_pixel_avg_vec(pool_az)



def cross_corr(imageA, imageB):

    from numpy import linalg, dot

    # Get average vector of image

    vec_A = get_pixel_avg_vec(imageA)

    vec_B = get_pixel_avg_vec(imageB)

    # Get L2 norm

    a_norm = linalg.norm(vec_A, 2)

    b_norm = linalg.norm(vec_B, 2)



    # Cosine Similarity

    sim = dot(vec_A / a_norm, vec_B / b_norm)

    return sim



cross_corr(pool_az, pool_ggl1), cross_corr(pool_az, pool_ggl2)

mse(pool_az, pool_ggl1), mse(pool_az, pool_ggl2)



from scipy.stsci.convolve import cross_correlate



cross_correlate(img_az, mode=FULL)
