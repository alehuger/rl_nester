from setuptools import find_packages, setup


setup(
    name="nester",
    version="1.0",
    description="",
    packages=find_packages(include=["nester", "nester.*"]),
    python_requires=">=3.10",
)
