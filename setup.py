from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name='cactus.kernels.matmul',  # accessible as cactus.kernels.matmul_cpp in Python
        sources=['cactus/kernels/matmul.cpp'],
        extra_compile_args=['-O3']
    )
]

setup(
    name="cactus-lib",
    version="0.1.5",
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
        "pybind11"
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
