from setuptools import setup, find_packages
import os

# Function to parse requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get version from __init__.py
version = {}
with open(os.path.join(this_directory, "neural_forest", "__init__.py")) as fp:
    exec(fp.read(), version)


setup(
    name='neural-forest',
    version=version['__version__'],
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    description='A Neural Forest implementation with TensorFlow and Keras',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/neural-forest', # Replace with your repo URL
    packages=find_packages(exclude=['examples*', 'tests*']), # find_packages will find 'neural_forest'
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha', # Or 4 - Beta, 5 - Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8', # Specify your Python version requirement
    keywords='neural forest, decision tree, ensemble, machine learning, tensorflow, keras',
    project_urls={ # Optional
        'Bug Reports': 'https://github.com/yourusername/neural-forest/issues',
        'Source': 'https://github.com/yourusername/neural-forest/',
    },
)