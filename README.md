# HYPer-spectral Image Augmentation (hypia)

The following is a Python package (available through the Python Package Manager (pip)) for the augmentation of hyper-spectral images for uses in deep learning.

### Why does this exist?

* Other image augmentation libraries often only accept RGB or grayscale images (with options for RGBA images usually too). However, in a scientific context, we often have images with more than 3/4 channels (so-called hyper-spectral imaging) and it seemed like a good idea to have an augmentation library which deals with all channels in parallel.
* Another problem is that some image augmentation frameworks can accept more than 3/4 channels but convert the data to unsigned 8-bit integers (see PyTorch's [torchvision](https://pytorch.org/docs/stable/torchvision/transforms.html)) which is damaging for scientific data where we care about the actual numbers of the data. This can lead to the images losing some of the features and relative contrasts of features which is important for our science.

I had a look at several image augmentation packages but none of them seemed to satisfy both of these criteria so here we are.

### Installation

There are two ways to install this package: either from pip or from source from GitHub.

From pip:

```bash
>>> pip install hypia
```

From GitHub:
```bash
>>> git clone https://github.com/rhero12/hypia
>>> cd hypia
>>> python setup.py install
```

### Documentation