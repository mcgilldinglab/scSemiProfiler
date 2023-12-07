from setuptools import setup, find_packages

setup(
    name="scSemiProfiler",
    version="1.0.0",
    description='scSemiProfiler package 1.0',
    author='Jingtao Wang',
    author_email = 'jingtao.wang@mail.mcgill.ca',
    url='https://github.com/mcgilldinglab/scSemiProfiler',
    entry_points={'console_scripts':['activeselect=scSemiProfiler.activeselect:main','scprocess=scSemiProfiler.scprocess:main','initsetup=scSemiProfiler.initsetup:main','scinfer=scSemiProfiler.scinfer:main']},
    #packages=['scSemiProfiler'],
    packages=find_packages(),
    classifiers=[
    'Programming Language :: Python :: 3.9'],
    install_requires=['numpy>= 1.26.2',
    'scanpy>= 1.9.6',
    'scipy>= 1.11.4',
    'anndata>= 0.10.3',
    'faiss-cpu>= 1.7.4',
    'torch>= 1.12.1',
    'scikit-learn>= 1.3.2',
    'pandas>= 2.1.3',
    'jax>= 0.4.19',
    'scvi-tools>= 1.0.4'],
)



