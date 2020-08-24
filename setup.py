from setuptools import setup, find_packages

# Go!
setup(
    # Module name (lowercase)
    name='pkpd',
    version='0.0.1dev0',

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@dtc.ox.ac.uk',

    # Packages to include
    packages=find_packages(include=('pkpd', 'pkpd.*')),

    # List of dependencies
    install_requires=[
        'jupyter==1.0.0',
        'myokit==1.30.6',
        'numpy>=1.8',
        'pandas>=0.24',
        'pints @ git+git://github.com/pints-team/pints.git#egg=pints',
        'plotly==4.8.1',
        'tqdm==4.46.1'
    ],
    dependency_links=[
     "git+git://github.com/pints-team/pints.git#egg=pints-0.2.2",
    ]
)
