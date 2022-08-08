from setuptools import find_packages, setup

setup(name='BaseTemplate',
        version='0.1',
        description='Unofficial implementation ',
        url='https://github.com/jonggyujang0123/Transformer-NLP',
        author='jonggyujang0123',
        author_email='jgjang0123@gmail.com',
        packages=  ['configs','models','utils', 'tools'],
        python_requires = '>=3.7')
