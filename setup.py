import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("adapt.yml", "r") as f:
    reqs = f.read()
reqs = reqs.split("- pip:")[1]
reqs = reqs.split("prefix")[0]
reqs = reqs.split("\n")
reqs = [x.replace("-", "", 1) for x in reqs if x.strip()]
reqs = [x.strip() for x in reqs if x.strip()]

setuptools.setup(
    name="PB speaker adaptation",
    version="0.1",
    packages=setuptools.find_packages(),
    long_description=long_description,
    install_requires=reqs,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
