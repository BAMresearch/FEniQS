import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FEniQS",
    version="0.1",
    author="Abbas Jafari",
    author_email="abbas.jafari@bam.de",
    description="A library for modelling static/quasi-static structural mechanics problems in FEniCS.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BAMresearch/FEniQS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
