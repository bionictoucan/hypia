import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hypia",
    version="0.0.2",
    author="John Armstrong",
    author_email="j.armstrong.2@research.gla.ac.uk",
    description="A package for doing hyper-spectral image augmentation for deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rhero12/Hypia",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)