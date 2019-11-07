from . import functionals as F

class Compose:
    '''
    This allows us to stack several transforms together to execute them in sequence.

    Parameters
    ----------
    transforms : list
        A list of the transforms to carry out. They will be carried out in the order passed to this function.
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Normalise:
    '''
    This class will create a normalising object with a certain mean and standard deviation.

    Parameters
    ----------
    mean : float
        The mean to subtract from the data.
    std : float
        The standard deviation to divide the data by.
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return F.normalise(img, self.mean, self.std)

class Resize:
    '''
    This class will resize the image.

    Parameters
    ----------
    size : int or tuple
        The size to resize the image to.
    interpolation_order : int, optional
        The order of the Spline interpolation to resize the image. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the resize. Default is False.
    channel_pos : str, optional
        The axis of the number of channels. Default is "first". Can also be "last".
    '''

    def __init__(self, size, interpolation_order=3, anti_aliasing=False, channel_pos="first"):
        self.size = size
        self.interp = interpolation_order
        self.aa = anti_aliasing
        self.cp = channel_pos

    def __call__(self, img):
        return F.resize(img, self.size, self.interp, self.aa, self.cp)

class Rescale:
    '''
    This class will create an object for rescaling images based on a specified scaling factor.

    Parameters
    ----------
    scale : float or tuple or list
        The factors to multiply the original axes by to rescale. Can be a single float (both axes scaled by same number) or a tuple/list (y, x) ordering of the scaling factors.
    interpolation_order : int, optional
        The order of the spline interpolation to use when resizing images. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the scaling of the images. Default is False.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, scale, interpolation_order=3, anti_aliasing=False, channel_pos="first"):
        self.scale = scale
        self.interp = interpolation_order
        self.aa = anti_aliasing
        self.cp = channel_pos

    def __call__(self, img):
        return F.rescale(img, self.scale, self.interp, self.aa, self.cp)

class Crop:
    '''
    This class will create an object for cropping images based on a specified box defined by the coordinates of the top left corner and the height and the width of the box.

    Parameters
    ----------
    tl : numpy.ndarray or tuple or list
        The coordinates of the top left corner of the crop in (y,x).
    height : int
        The height in pixels of the crop.
    width : int
        The width in pixels of the crop.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, tl, height, width, channel_pos="first"):
        self.tl = tl
        self.h = height
        self.w = width
        self.cp = channel_pos

    def __call__(self, img):
        return F.crop(img, self.tl, self.h, self.w, self.cp)

class HorizontalFlip:
    '''
    This class creates an object that will horizontally flip an image.

    Parameters
    ----------
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, channel_pos="first"):
        self.cp = channel_pos

    def __call__(self, img):
        return F.hflip(img, self.cp)

class VerticalFlip:
    '''
    This class creates an object that will vertically flip an image.

    Parameters
    ----------
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, channel_pos="first"):
        self.cp = channel_pos

    def __call__(self, img):
        return F.vflip(img, self.cp)

class Rotate:
    '''
    This class creates an object that will rotate an image by an angle theta anti-clockwise with respect to the x axis.

    Parameters
    ----------
    angle : float
        The angle to rotate the image by (radians).
    reshape : bool, optional
        Whether or not to reshape the rotated array such that the entire image is still in the field-of-view. Default is False.
    interpolation_order : int, optional
        The order of the spline interpolation to use. Default is 3.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, angle, reshape=False, interpolation_order=3, channel_pos="first"):
        self.angle = angle
        self.reshape = reshape
        self.interp = interpolation_order
        self.cp = channel_pos

    def __call__(self, img):
        return F.rotate(img, self.angle, self.reshape, self.interp, self.cp)

class Erase:
    '''
    This class creates an object that will erase a box of pixels from an image defined by the coordinates of the top left corner of the box to be erased along with the height and width of this box.

    Parameters
    ----------
    tl : numpy.ndarray or tuple or list
        The top left corner of the box to erase in (y,x) format.
    height : int
        The height of the box to be erased in pixels.
    width : int
        The width of the box to be erased in pixels.
    val : float
        The value to replace the erased box with.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, tl, height, width, val, channel_pos="first"):
        self.tl = tl
        self.h = height
        self.w = width
        self.v = val
        self.cp = channel_pos

    def __call__(self, img):
        return F.erase(img, self.tl, self.h, self.w, self.v, self.cp)

class Shear:
    '''
    This class creates an object to shear an image in the x direction i.e. transformation matrix

    .. math::
        S = \\begin{pmatrix}
               1 & -\sin \\beta & 0 \n
               0 & \cos \\beta & 0 \n
               0 & 0 & 1
            \end{pmatrix}

    where :math:`\\beta` is the shearing angle anti-clockwise with respect to the x-axis.

    Parameters
    ----------
    angle : float
        The angle of the shearing (radians).
    interpolation_order : int, optional
        The order of the spline interpolation to use. Default is 3.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, angle, interpolation_order=3, channel_pos="first"):
        self.angle = angle
        self.interp = interpolation_order
        self.cp = channel_pos

    def __call__(self, img):
        return F.shear(img, self.angle, self.interp, self.cp)

class AffineTransform:
    '''
    This class creates an object to apply a general affine transform to an image using the implementation from `scikit-image <https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.AffineTransform>`_.

    .. math::
        A = \\begin{pmatrix}
               s_{x} \cos \\theta & - s_{y} \sin (\\theta + \\beta) & T_{x} \n
               s_{x} \sin \\theta & s_{y} \cos (\\theta + \\beta) & T_{y} \n
               0 & 0 & 1
            \\end{pmatrix}

    where :math:`s_{x}, ~s_{y}` are the scales in the x and y directions, respectively. :math:`\\theta` is the angle of rotation anti-clockwise from the x direction. :math:`\\beta` is the angle of shear anti-clockwise from the x direction. :math:`T_{x}, ~T_{y}` are the translations in the x and y directions, respectively.

    Parameters
    ----------
    scales : numpy.ndarray or tuple or list, optional
        The scaling factor in the x and y directions expressed in (s_x, s_y). Default is None.
    rotate : float, optional
        The rotation angle anti-clockwise with respect to the x axis (in radians). Default is None.
    shear : float, optional
        The shear angle anti-clockwise with respect to the x axis (in radians). Default is None.
    translate : numpy.ndarray or tuple or list, optional
        The translations in the x and y directions expressed in (T_x, T_y). Default is None.
    interpolation_order : int, optional
        The order of the spline interpolation to use in the affine transformation. Default is 3.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, scales=None, rotate=None, shear=None, translate=None, interpolation_order=3, channel_pos="first"):
        self.scales = scales
        self.rotate = rotate
        self.shear = shear
        self.translate = translate
        self.interp = interpolation_order
        self.cp = channel_pos

    def __call__(self, img):
        return F.affine_transform(img, self.scales, self.rotate, self.shear, self.translate, self.interp, self.cp)

class Zoom:
    '''
    This class with create an object to zoom into an image to a certain box defined by the top left corner's coordinates of the box and the height and width of the box. This is a combination of the crop and resize transforms.

    Parameters
    ----------
    tl : numpy.ndarray or tuple or list
        The coordinates of the top left corner of the box to zoom to in (y,x).
    height : int
        The height of the box to zoom to in pixels.
    width : int
        The width of the box to zoom to in pixels.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    interpolation_order : int, optional
        The order of the spline interpolation to use in the resize. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the resize. Default is False.
    '''

    def __init__(self, tl, height, width, channel_pos="first", interpolation_order=3, anti_aliasing=False):
        self.tl = tl
        self.h = height
        self.w = width
        self.cp = channel_pos
        self.interp = interpolation_order
        self.aa = anti_aliasing

    def __call__(self, img):
        return F.zoom(img, self.tl, self.h, self.w, self.cp, self.interp, self.aa)

class Stretch:
    '''
    This class creates an object that stretches an image in the y direction. This is a combination of the shear and resize transforms.

    Parameters
    ----------
    angle : float
        The shearing angle (radians).
    interpolation_order : int, optional
        The order of the spline interpolation to use in both the shear and the resize. Default is 3.
    anti_aliasing : bool, optional
        Whether or not to use anti-aliasing in the resize. Default is False.
    channel_pos : str, optional
        Whether or not the spectral axis (i.e. the number of channels in the image) is at the beginning of the array or the end. Default is "first". Can be either "first" or "last".
    '''

    def __init__(self, angle, interpolation_order=3, anti_aliasing=False, channel_pos="first"):
        self.angle = angle
        self.interp = interpolation_order
        self.aa = anti_aliasing
        self.cp = channel_pos

    def __call__(self, img):
        return F.stretch(img, self.angle, self.interp, self.aa, self.cp)