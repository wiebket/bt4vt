import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    packages=setuptools.find_packages(where="bt4vt"),
    python_requires=">=3.6",
)