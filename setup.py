from setuptools import setup, find_packages
from typing import List

# Constant for editable installs
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements file and returns a list of dependencies.
    Removes '-e .' if present, which is used for editable installs.
    
    Args:
        file_path (str): Path to the requirements file.

    Returns:
        List[str]: List of package dependencies.
    """
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters and strip whitespace
        requirements = [req.strip() for req in requirements if req.strip()]

        # Remove editable install flag if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='Samvad',
    version='1.0.0',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A PyTorch-based Seq2Seq model with attention for ontinuous Indian Sign Language Translator System.',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    author='Decodians',
    author_email='samvad@decodians.com',
    url='https://github.com/zaibreyaz/Samvad.git',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.11.0',
    include_package_data=True,
    keywords=["sign language", "seq2seq", "pytorch", "deep learning", "CSLR"],
)
