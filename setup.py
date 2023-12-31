from setuptools import setup, find_packages

setup(
    name="scSemiProfiler",
    version="1.0.0",
    description='scSemiProfiler package 1.0',
    author='Jingtao Wang',
    author_email = 'jingtao.wang@mail.mcgill.ca',
    url='https://github.com/mcgilldinglab/scSemiProfiler',
    entry_points={'console_scripts':['activeselect=scSemiProfiler.representative_selection:main','scprocess=scSemiProfiler.singlecell_process:main','initsetup=scSemiProfiler.initial_setup:main','scinfer=scSemiProfiler.inference:main','get_eg_representatives=scSemiProfiler.get_eg_representatives:main']},
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
    'igraph>=0.9.9',
    'gseapy>=1.0.4',                  
    'scvi-tools>= 1.0.4'],
)



