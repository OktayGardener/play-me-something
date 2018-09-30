from setuptools import find_packages
from setuptools import setup

setup(
    name='pms',
    version='1.0',
    packages=find_packages(),
    package_data={'pms': ['config/deep_model.yaml']},
    include_package_data=True,
    install_requires=["google-cloud-storage"]
)
