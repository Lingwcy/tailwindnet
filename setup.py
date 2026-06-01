import os
import re
from setuptools import find_packages, setup


def parse_version():
    init_path = os.path.join(os.path.dirname(__file__), 'mmseg_plugin', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    return m.group(1) if m else '0.1.0'


setup(
    name='tailwindnet',
    version=parse_version(),
    description='TailwindNet: efficient multimodal backbones for semantic segmentation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='TailwindNet authors',
    license='Apache License 2.0',
    packages=find_packages(exclude=('configs', 'tools', 'resources', 'resources.*')),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.13',
        'torchvision',
        'mmengine>=0.7',
        'mmcv>=2.0.0',
        'mmsegmentation>=1.0',
        'einops',
        'antialiased-cnns',
        'timm',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False,
)
