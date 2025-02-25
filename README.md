[![Unit tests on multiple python versions](https://github.com/DavAug/ErlotinibGefitinib/workflows/Unit%20tests%20(python%20versions)/badge.svg)](https://github.com/DavAug/ErlotinibGefitinib/actions)
[![Unit tests on multiple operating systems](https://github.com/DavAug/ErlotinibGefitinib/workflows/Unit%20tests%20(OS%20versions)/badge.svg)](https://github.com/DavAug/ErlotinibGefitinib/actions)
[![codecov](https://codecov.io/gh/DavAug/ErlotinibGefitinib/branch/master/graph/badge.svg)](https://codecov.io/gh/DavAug/ErlotinibGefitinib)

# ErlotinibGefitinib

In this project the pharmacokinetic and pharmacodynamic (PKPD) properties of Erlotinib and Gefitinib are investigated. In particular, we re-analyse a study published in 2016 [1], which aimed at building a translational PKPD model from mice data to predict tumour stasis inducing dosing regimens in humans. To this end, we will in first instance focus on reproducing the modelling results. In a second step, we will question the modelling choices and try to
introduce a robust and reproducible PKPD modelling approach.

The analysis of the data, and the modelling of the PKPD is organised in Jupyter notebooks. We start with modelling the control group, 
and gradually build up the inference of the Erlotinib/Gefitinib specific parameters. This will allow us to explore efficacious dosing regimens across individuals *in silico*.

## Reproducibility

Each notebook is well documented, self-contained, and contains interactive visualisations of the data and modelling results. An out-of-the-box solution to view and run the interactive notebooks is kindly provided by [nbviewer](https://nbviewer.jupyter.org/) and [binder](https://mybinder.org/). Please note that binder takes a couple of minutes (~3min) to set up the server to enable you to run the notebooks. Alternatively you are welcome to clone and install the repository to run the analysis locally, see INSTALL.md.

The data as well as the models used in the analysis are made publically available in this repository, see DATA.md and MODELS.md. In particular, the models are made accessible in SBML file format, such that they may be used in other modelling projects.

## Notebook references

- [Overview](https://nbviewer.jupyter.org/github/DavAug/ErlotinibGefitinib/blob/master/notebooks/overview/overview.ipynb)
- Lung cancer (LXF A677)
    - Control Growth
    - Treatment with Erlotinib
    - Treatment with Gefitinib
- Vulva cancer (VXF A431)
    - Control Growth
    - Treatment with Erlotinib
    - Treatment with Gefitinib
- Model selection: Published versus derived model
- Summary
    
## Bibliography

- <a name="ref1"> [1] </a> Eigenmann et. al., Combining Nonclinical Experiments with Translational PKPD Modeling to Differentiate Erlotinib and Gefitinib, Mol Cancer Ther (2016)
