import setuptools

def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(name='toppicker',
    version='0.1',
    description='ML based framework and application for identifying formation top from well log data',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/haizadtarik/formation-top-picker',
    author='haizad, russel, ashraf, CY and Hadi',
    license="MIT",
    install_requires=required,
    packages=setuptools.find_packages(),
    zip_safe=False)