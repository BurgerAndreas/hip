import setuptools


setuptools.setup(
    name="hip",
    version="1.0.0",
    # packages=setuptools.find_packages(),
    packages=[
        "hip",
        "nets",
        "ocpmodels",
        "alphanet",
        "leftnet",
        "recipes",
        "scripts",
    ],
)
