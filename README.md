# ErlotinibGefitinib

In this project the PKPD properties of Erlotinib and Gefitinib are investigated. In particular, we re-analyse a study published in 2016 [1],
which aimed at building a translational PKPD model from mice data to predict tumour stasis inducing dosing regimens in humans. In this project
we will in first instance focus on reproducing the modelling results in [1]. In a second step, we will question the modelling choices and try to
introduce a robust and reproducible PKPD modelling approach.

The analysis of the data, and the modelling of the PKPD is organised in Jupyter notebooks. We start with modelling the control group, 
and gradually build up the inference of the Erlotinib/ Gefitinib specfic parameters.

## Notebook structure

- [Overview](https://github.com/DavAug/ErlotinibGefitinib/blob/master/notebooks/overview/overview.ipynb)
- [Growth modelling in absence of treatment](https://nbviewer.jupyter.org/github/DavAug/ErlotinibGefitinib/blob/master/notebooks/control_growth/data_preparation.ipynb)
    - [Data](https://nbviewer.jupyter.org/github/DavAug/ErlotinibGefitinib/blob/master/notebooks/control_growth/data_preparation.ipynb)
    - [Pooled model](https://github.com/DavAug/ErlotinibGefitinib/blob/master/control_growth_analysis.ipynb)
    - Unpooled model
    - Hierarchical model
- [Growth modelling under Erlotinib treatment]
    - Data
    - Pooled model
    - Hierarchical model
- [Growth modelling under Gefitinib treatment]
    - Data
    - Pooled model
    - Hierarchical model
    
## Bibliography

- <a name="ref1"> [1] </a> Eigenmann et. al., Combining Nonclinical Experiments with Translational PKPD Modeling to Differentiate Erlotinib and Gefitinib, Mol Cancer Ther (2016)
