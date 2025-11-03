from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="clvtnet",
    version="1.0.0",
    author="Research Team",
    author_email="research@university.edu",
    description="CLVTNet: Convolutional-LSTM Vision Transformer for EEG Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/clvtnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "wandb>=0.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "clvtnet-train=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "clvtnet": ["configs/*.yaml", "data/*.mat"],
    },
)