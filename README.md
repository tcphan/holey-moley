## holey-moley: A Topological Data Analysis (TDA) Package
The goal of this project is to explore various creative applications of TDA in order to give users a "hole" new perspective of their data (pun intended :smirk:) and inspire greater usage and discussion of these methods in data science.

Each Jupyter notebook under the `notebooks` folder walks through a concrete example of a TDA use case with code and interpretations provided. The source scripts under the `tda` folder are designed to be flexible so that users can readily apply them out-of-the-box to their own custom data as well as scalable enough to be ran efficiently on large datasets.

## Overview of Uses Cases:

Below is list of the currently available example use cases:

  1. [exploring_sdoh_w_tda.ipynb](https://github.com/tcphan/holey-moley/blob/main/notebooks/exploring_sdoh_w_tda.ipynb) : Applying TDA to visualize high-dimensional, structural differences in social determinants of health outcomes between states.
  2. [feature_drift_detection.ipynb](https://github.com/tcphan/holey-moley/blob/main/notebooks/feature_drift_detection.ipynb) : Measuring distances between persistence diagrams to detect feature drift in high-dimensional datasets.
  3. [health_risk_topological_matching.ipynb](https://github.com/tcphan/holey-moley/blob/main/notebooks/health_risk_topological_matching.ipynb) : Applying Mapper-based TDA pipeline (UMAP + binning + clustering) to build cohorts of similar high-risk patients for the purposes of A/B testing new health interventions.
