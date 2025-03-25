from setuptools import setup, find_packages

setup(
    name="cactus-lib",
    version="0.1.4",
    description="Framework for fine-tuning LLMs on the Cactus Compute platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Cactus Compute, Inc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "transformers",
        "platformdirs",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)