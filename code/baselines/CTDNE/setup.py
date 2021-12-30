from distutils.core import setup

setup(
    name='CTDNE',
    packages=['CTDNE'],
    version='0.0.1',
    description='Implementation of the CTDNE algorithm.',
    author='Uriel Singer',
    author_email='urielsinger@gmail.com',
    license='MIT',
    url='https://github.com/urielsinger/CTDNE',
    install_requires=[
        'networkx',
        'gensim',
        'numpy',
        'tqdm',
        'joblib'
    ],
    keywords=['machine learning', 'embeddings', 'temporal'],
)