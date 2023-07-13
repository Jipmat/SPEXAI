from setuptools import setup

setup(
    name='spexai',
    version='0.0.1',    
    description='A neural network emulator for SPEX',
    url='https://github.com/Jipmat/SPEXAI',
    author='Jip Matthijsse',
    author_email='',
    license='TBD',
    packages=['spexai'],
    install_requires=['numpy',
                      'pandas',
#                      'pytorch',
                      'scikit-learn',
                      'matplotlib',
#                      'emcee',
                      'scipy',
                      'seaborn',
                      'astropy'                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: ',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)

