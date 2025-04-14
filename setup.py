from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = [
    CppExtension(
        name='sllm.ops.matmul', 
        sources=['sllm/ops/matmul.cpp'],
        extra_compile_args=['-O3']
    )
]

setup(
    name="sllm-lib",
    version="0.0.1",
    description="Super Lazy Language Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Henry Ndubuaku",
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
