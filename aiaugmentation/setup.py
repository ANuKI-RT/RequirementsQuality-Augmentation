from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Software Requirements Augmentation/Creation"
LONG_DESCRIPTION = "Package for augmenting and creating software requirements, utilizing the power of gpt and fairseq."

setup(
    name="aiaugmentation",
    version=VERSION,
    author="Robin Korfmann",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
)