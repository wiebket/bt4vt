import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sveva",
    version="0.0.1",
    author="Wiebke Toussaint",
    author_email="w.toussaint@tudelft.nl",
    description="Speaker Verification model EVAluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wiebket/sveva",
    license="GPL-3.0-or-later"
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
    ],
    package_dir={"": "sveva"},
    packages=setuptools.find_packages(where="sveva"),
    python_requires=">=3.6",
)