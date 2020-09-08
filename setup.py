from setuptools import setup, find_packages

# Go!
setup(
    # Module name
    name='pkpd',
    version='0.0.1dev0',

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@dtc.ox.ac.uk',

    # Packages and data to include
    packages=find_packages(include=('pkpd', 'pkpd.*')),
    package_data={'pkpd': ['*.xml']},
    include_package_data=True,

    # List of dependencies
    install_requires=[
        'jupyter==1.0.0',
        'myokit>=1.31',
        'numpy>=1.8',
        'pandas>=0.24',
        'pints>=0.3',
        'plotly==4.8.1',
        'tqdm==4.46.1'
    ],
)
