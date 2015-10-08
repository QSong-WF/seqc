__author__ = 'Ambrose J. Carr'

from setuptools import setup

setup(name='seqc',
      version='0.1',
      description='Single Cell Sequencing Processing and QC Suite',
      author='Ambrose J. Carr',
      author_email='mail@ambrosejcarr.com',
      package_dir={'': 'src'},
      # note: requires numpy > 1.10.0
      packages=['seqc'],
      install_requires=[
          'numpy>=1.10.0',
          'pandas>=0.16.0',
          'matplotlib>=1.4.3',
          'seaborn',
          'scipy>=0.14.0',
          'boto3'],
      scripts=['src/scripts/SEQC', 'src/scripts/PROCESS_BARCODES',
               'src/scripts/TEST_BARCODES'],
      )

