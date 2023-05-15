from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
  name = 'dualopt',
  packages = find_packages(exclude=['examples']),
  version = '0.1.7',
  license='MIT',
  description = 'Dual Optimizer Training',
  long_description=long_description,
  long_description_content_type = 'text/markdown',
  author = 'Pranav Jeevan',
  author_email = 'pranav13phoenix@gmail.com',
  url = 'https://github.com/pranavphoenix/DualOpt',
  keywords = [
    'artificial intelligence',
    'machine learning',
    'optimizer',
    'training',
    'image classification'
    'semantic segmentation',
    'image super-resolution'
  ],
  install_requires=[
    'torch',
    'torchvision',
    'torchmetrics',
    'numpy',
    'lion_pytorch'
  ],
  
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)