import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="evodenss",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Adriano Vinhas",
    author_email="avinhas@dei.uc.pt",
    description="A Python library that performs Neuro-Evolution by evolving the structure of the networks and optimiser aspects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adrianovinhas/fast-denser-adriano",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.9',
    install_requires=requirements,
)