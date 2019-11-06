import numpy as np
import scipy.ndimage as ndi
from skimage.transform import warp, AffineTransform

def normalise(img, mean, std):
    '''
    Normalise an image with a certain mean and standard deviation. Can be different for different bands.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be normalised.
    mean : float or numpy.ndarray
        Either a float to be the mean for all bands or a numpy.ndarray with the mean for each band.
    std : float or numpy.ndarray
        The same as for the means but with standard deviations instead.

    Returns
    -------
    norm_img : numpy.ndarray
        The normalised image.
    '''

    norm_img = (img - mean) / std

    return norm_img

def resize(img, size, interpolation_order=3, anti_aliasing=False, channel_pos="first"):
    '''
    This is a function for resizing the image to a different size.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be resize.
    size : int or tuple
        The size to resize the image to.
    interpolation_order : int, optional
        The order of the Spline interpolation to resize the image. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the resize. Default is False.
    channel_pos : str, optional
        The axis of the number of channels. Default is "first". Can also be "last".

    Returns
    -------
    resize_img : numpy.ndarray
        The resize of the image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)
    input_shape = img.shape
    if type(size) == int: 
        output_shape = [size, size, img.shape[-1]]
    elif type(size) == (tuple or list) and len(size) == 2:
        output_shape = [*size, img.output_shape[-1]]
    elif type(size) == (tuple or list) and len(size) == 3:
        output_shape = [*size]

    assert(input_shape[-1] == output_shape[-1])

    factors = np.asarray(input_shape) / np.asarray(output_shape) #this tells us the scaling of the resize to the image in each dimension

    if anti_aliasing:
        anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)

        img = ndi.gaussian_filter(img, anti_aliasing_sigma)

    #this defines the corners for the resized image to define the inverse mapping
    src_corners = np.array([[0, 0], [0, output_shape[0]-1], [output_shape[1]-1, output_shape[0]-1]])
    dst_corners = np.zeros_like(src_corners)

    #a pixel's position is technically its centre e.g. (0, 0) -> (0.5, 0.5)
    dst_corners[:,0] = factors[1] * (src_corners[:,0] + 0.5) - 0.5
    dst_corners[:,1] = factors[0] * (src_corners[:,1] + 0.5) - 0.5

    tform = AffineTransform()
    tform.estimate(src_corners, dst_corners)

    tform.params[2] = (0, 0, 1)
    tform.params[0,1] = 0
    tform.params[1,0] = 0
    #the above makes sure the transform does not shear the image

    out = warp(img, tform, output_shape=output_shape, order=interpolation_order, preserve_range=True)

    if channel_pos == "first":
        #redo the transformation so the channels are in the right axis
        out = np.moveaxis(out, -1, 0)

    return out

def rescale(img, scale, interpolation_order=3, anti_aliasing=False, channel_pos="first"):
    '''
    This function uses the resize function defined above to scale images depending on scale factors passed as arguments to this function.
    
    Parameters
    ----------
    img : numpy.ndarray
        The image to be rescaled.
    scale : float or tuple or list
        The factors to multiply the original axes by to rescale. Can be a single float (both axes scaled by same number) or a tuple/list with (y, x) ordering of the scaling factors.
    interpolation_order : int, optional
        The order of the spline interpolation to use when resizing images. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the scaling of the images. Default is False.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".

    Returns
    -------
    out : numpy.ndarray
        The rescaled image.
    '''

    if channel_pos == "first":
        input_shape = img.shape[1:]
    else:
        input_shape = img.shape[:2]
    size = np.round(scale * input_shape)
    #we extract just the y and x axes of the images and multiply this by the scales to get the size of reshaped image

    out = resize(img, size, interpolation_order=interpolation_order, anti_aliasing=anti_aliasing, channel_pos=channel_pos)

    return out


def crop(img, tl, height, width, channel_pos="first"):
    '''
    This function crops the image to a defined size.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be cropped.
    tl : numpy.ndarray or tuple or list
        The coordinates of the top left corner of the crop.
    height : int
        The height in pixels of the crop.
    width : int
        The width in pixels of the crop.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    cropped : numpy.ndarray
        The cropped image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)
    
    if tl[0]+height > img.shape[0] or tl[1]+width > img.shape[1]:
        raise IndexError("Cannot crop to a size bigger than the image!")
    cropped = img[tl[0]:tl[0]+height,tl[1]:tl[1]+width]

    if channel_pos == "first":
        cropped = np.moveaxis(cropped, -1, 0)

    return cropped

def hflip(img, channel_pos="first"):
    '''
    This function horizontally flips the image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be flipped horizontally.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    flipped : numpy.ndarray
        The horizontally flipped image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    flipped = np.fliplr(img)

    if channel_pos == "first":
        flipped = np.moveaxis(flipped, -1, 0)

    return flipped

def vflip(img, channel_pos="first"):
    '''
    This function vertically flips the image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be flipped vertically.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    flipped : numpy.ndarray
        The vertically flipped image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    flipped = np.flipud(img)

    if channel_pos == "first":
        flipped = np.moveaxis(flipped, -1, 0)

    return flipped

def rotate(img, angle, reshape=False, interpolation_order=3, channel_pos="first"):
    '''
    This function is essentially a Hypia wrapper for `scipy.ndimage.rotate` but with the option for what channel position the image has. This rotates the image by an angle theta anti-clockwise with respect to the x axis.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be rotated.
    angle : float
        The angle to rotate the image by (radians).
    reshape : bool, optional
        Whether or not to reshape the rotated array such that the entire image is still in the field-of-view. Default is False.
    interpolation_order : int, optional
        The order of the spline interpolation to use.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    rotated : numpy.ndarray
        The rotated image by angle theta with respect to the x axis.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)
    
    rotated = ndi.rotate(img, np.rad2deg(angle), reshape=reshape, order=interpolation_order)

    if channel_pos == "first":
        rotated = np.moveaxis(rotated, -1, 0)

    return rotated

def erase(img, tl, height, width, val, channel_pos="first"):
    '''
    This function will erase a section of the of the image defined by the top left corner with the height and the width of the box.

    Parameters
    ----------
    img : numpy.ndarray
        The image to have an area erased.
    tl : numpy.ndarray or tuple or list
        The top left corner of the image in (y, x) format.
    height : int
        The height of the box to be erased in pixels.
    width : int
        The width of the box to be erased in pixels.
    val : float
        The value to replace the erased box with.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    img : numpy.ndarray
        The image with a box erased.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    img[tl[0]:tl[0]+height,tl[1]:tl[1]+width] = val

    if channel_pos == "first":
        img = np.moveaxis(img, -1, 0)

    return img

def shear(img, angle, interpolation_order=3, channel_pos="first"):
    '''
    This function shears the image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be sheared.
    angle : float
        The angle of the shearing (radians).
    interpolation_order : int, optional
        The order of the spline interpolation to use.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    sheared : numpy.ndarray
        The sheared image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    tform = AffineTransform(shear=angle)

    sheared = warp(img, tform.inverse, order=interpolation_order, preserve_range=True)

    if channel_pos == "first":
        sheared = np.moveaxis(sheared, -1, 0)

    return sheared

def affine_transform(img, scales=None, rotate=None, shear=None, translate=None, interpolation_order=3,channel_pos="first"):
    '''
    A general affine transform using the implementation for scikit-image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to apply the transform to.
    scales : numpy.ndarray or tuple or list, optional
        The scaling factor in the x and y directions expressed in (s_x, s_y). Default is None.
    rotate : float, optional
        The rotation angle anti-clockwise with respect to the x axis. Default is None.
    shear : float, optional
        The shear angle anti-clockwise with respect to the x axis. Default is None.
    translate : numpy.ndarray or tuple or list, optional
        The translations in the x and y directions expressed in (T_x, T_y). Default is None.
    interpolation_order : int, optional
        The order of spline interpolation to use in the affine transformation. Default is 3.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    transformed : numpy.ndarray
        The result of applying the affine transformation to the original image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    tform = AffineTransform(scale=scales, rotation=rotate, shear=shear, translation=translate)

    transformed = warp(img, tform.inverse, order=interpolation_order, preserve_range=True)

    if channel_pos == "first":
        transformed = np.moveaxis(transformed, -1, 0)

    return transformed

def zoom(img, tl, height, width, channel_pos="first", **resize_kwargs):
    '''
    This is a function to zoom on a particular part of the image which is done by cropping to a certain box and resizing the image to the resolution of the image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be zoomed.
    tl : numpy.ndarray or tuple or list
        The coordinates of the top left corner of the box to zoom to in (y,x).
    height : int
        The height of the box to zoom to in pixels.
    width : int
        The width of the box to zoom to in pixels.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    zoomed : numpy.ndarray
        Original image zoomed to a particular box within the original image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    zoomed = crop(img, tl, height, width, channel_pos="last")

    zoomed = resize(zoomed, size=img.shape, channel_pos="last", **resize_kwargs)

    if channel_pos == "first":
        zoomed = np.moveaxis(zoomed, -1, 0)

    return zoomed

def stretch(img, angle, interpolation_order=3, anti_aliasing=False,  channel_pos="first"):
    '''
    This function stretches an image in the x direction by applying a shear followed by a resize to the resolution of the image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be stretched.
    angle : float
        The shearing angle (radians).
    interpolation_order : int, optional
        The order of the spline interpolation to use in both the shear and the resize. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the resize. Default is False.
    channel_pos : str, optional
        The position of the spectral axis e.g. the number of channels. Default is "first". Can be either "first" or "last".

    Returns
    -------
    stretched : numpy.ndarray
        The stretched form of the original image.
    '''

    if channel_pos == "first":
        img = np.moveaxis(img, 0, -1)

    stretched = shear(img, angle, interpolation_order=interpolation_order, channel_pos="last")

    stretched = resize(stretched, size=img.shape, interpolation_order=interpolation_order, anti_aliasing=anti_aliasing, channel_pos="last")

    if channel_pos == "first":
        stretched = np.moveaxis(stretched, -1, 0)

    return stretched