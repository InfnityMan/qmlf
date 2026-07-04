import re
from pathlib import Path

from setuptools import setup, find_namespace_packages

here = Path(__file__).parent

long_description = (here / "README.md").read_text(encoding="utf-8")

requirements = [
    line.strip()
    for line in (here / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

version = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    (here / "qmlf" / "__init__.py").read_text(encoding="utf-8"),
    re.M
).group(1)

setup(
    name="qmlf",
    version=version,
    description="A quantum machine learning framework built on Qiskit, PyTorch and scikit-learn.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="InfnityMan",
    url="https://github.com/InfnityMan/qmlf",
    license="Apache-2.0",
    packages=find_namespace_packages(include=("qmlf", "qmlf.*")),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qmlf = qmlf.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research"
    ]
)
