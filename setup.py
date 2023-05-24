import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name="Speaker Adaptation",
    version="1.0.0",
    packages=setuptools.find_packages(),
    long_description=long_description,
    install_requires=reqs,
    long_description_content_type="text/markdown",
    author="Nicolo' Brandizzi",
    author_email="brandizzi@diag.uniroma1.it",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
