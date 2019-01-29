from setuptools import setup, find_packages

setup(
    name='layout_analysis',
    version='0.0.1',
    packages=find_packages(),
    license='LGPL-v3.0',
    long_description=open("README.md").read(),
    include_package_data=True,
    author="Alexander Hartelt, Christoph Wick",
    author_email="christoph.wick@informatik.uni-wuerzburg.de",
    url="https://gitlab2.informatik.uni-wuerzburg.de/OMMR4all/ommr4all-layout-analysis.git",
    download_url='https://gitlab2.informatik.uni-wuerzburg.de/OMMR4all/ommr4all-layout-analysis.git',
    install_requires=open("requirements.txt").read().split(),
    extras_require={
        'tf_cpu': ['tensorflow>=1.6.0'],
        'tf_gpu': ['tensorflow-gpu>=1.6.0'],
    },
    keywords=['OMR', 'layout analysis', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
)
