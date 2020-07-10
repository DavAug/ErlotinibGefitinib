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
    packages=find_packages(include=('pkpd','pkpd.*')),

    # List of dependencies
    install_requires=[
        'jupyter',
        'numpy>=1.8',
        'pandas',
        'plotly',
        'myokit>=1.29',
        'pints @ git+git://github.com/pints-team/pints.git#egg=pints'
    ],
    dependency_links=[
     "git+git://github.com/pints-team/pints.git#egg=pints-0.2.2",
    ]
)