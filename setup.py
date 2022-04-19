import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='PB speaker adaptation',
    version='0.1',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
