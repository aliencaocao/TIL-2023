from setuptools import find_packages, setup

setup(name='TranReID',
      version='1.0.0',
      description='TransReID: Transformer-based Object Re-Identification',
      author='xxx',
      author_email='xxx',
      url='xxx',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'h5py', 'opencv-python', 'yacs', 'timm'
          ],
      packages=find_packages(),
      keywords=[
          'Pure Transformer',
          'Object Re-identification'
      ])
