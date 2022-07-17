import setuptools
import os
from pathlib import Path
from glob import glob

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
example_dir = os.path.join(str(Path.home()), 'bias_tests_4_voice_tech','example')
os.makedirs(example_dir, exist_ok=True)

setuptools.setup(
    name="bt4vt",
    version="0.1",
    author="Wiebke Toussaint, Anna Leschanowsky",
    author_email="w.toussaint@tudelft.nl",
    description="Bias Test for Voice Technologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wiebket/bt4vt",
    license="GPL-3.0-or-later",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
    ], 
    include_package_data=True,
    data_files=[(example_dir, 
                 [os.path.join(d, x)
                  for d, dirs, files in os.walk('example')
                  for x in files if '.ipynb_checkpoints' not in d and 'figures' not in d
                 ])],
    packages=setuptools.find_packages(where="bt4vt"),
    python_requires=">=3.6",
)