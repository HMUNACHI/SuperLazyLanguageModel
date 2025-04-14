from setuptools import setup, Extension
import numpy as np
import os

# Detect OpenMP availability
def check_openmp():
    import tempfile
    import shutil
    from distutils.ccompiler import new_compiler
    
    cc = new_compiler()
    tmpdir = tempfile.mkdtemp()
    
    try:
        # Write a test file
        test_file = os.path.join(tmpdir, 'test_openmp.c')
        with open(test_file, 'w') as f:
            f.write('#include <omp.h>\nint main() { return 0; }\n')
        
        # Try to compile
        objects = cc.compile([test_file], output_dir=tmpdir, 
                             extra_postargs=['-fopenmp'])
        return True
    except:
        return False
    finally:
        shutil.rmtree(tmpdir)

extra_compile_args = []
extra_link_args = []

if check_openmp():
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
else:
    print("OpenMP not available, building without parallel support")

# Define the extension module
matmul_ext = Extension(
    'sllm.ops._matmul_ext',
    sources=['sllm/ops/matmul_ext.cpp'],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='sllm',
    version='0.1.5',
    description='Super Lazy Language Model Library',
    author='Henry',
    packages=['sllm', 'sllm.nn', 'sllm.ops'],
    ext_modules=[matmul_ext],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.0',
    ],
)
