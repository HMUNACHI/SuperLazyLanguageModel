from setuptools import setup, find_packages

setup(
	name="cactus",
	version="0.1.0",
	description="Framework for fine-tuning LLMs on the Cactus Compute platform",
	long_description=open("README.md").read(),
	long_description_content_type="text/markdown",
	author="Cactus Compute, Inc",
	packages=find_packages(),
	install_requires=[
		"datasets==3.4.1",
		"numpy==2.0.2",
		"pandas==2.2.3",
		"requests==2.32.3",
		"scipy==1.15.2",
		"torch==2.6.0",
		"transformers==4.49.0"
	],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
