This repo is cloned from https://github.com/bioshape-lab/dyn. Please check their README file for environment setup.

The following three notebooks consists of the implementation for three tasks for DUNN/DLM8 cell lines.

- [explore_quantitle.ipynb](https://github.com/wxli0/dyn/blob/main/dyn/notebooks/explore_quantitle.ipynb): Visualization of different quantiles.

- [explore_histogram.ipynb](https://github.com/wxli0/dyn/blob/main/dyn/notebooks/explore_histogram.ipynb): Histogram of distances to the mean.

- [explore_dimension_reduction.ipynb](https://github.com/wxli0/dyn/blob/main/dyn/notebooks/explore_dimension_reduction.ipynb): Dimensionality reduction.

[GSI2023_SI.pdf](https://github.com/wxli0/dyn/blob/main/dyn/manuscript/GSI2023_SI.pdf) contains supplementary figures to our main manuscript, showing results of our study on the DLM8 cell line and additional results for the DUNN cell line.

For [Geometry Information](https://link.springer.com/collections/cadahjefhd) submission, the new dataset of breast cancer image is in [breast_cancer](https://github.com/wxli0/dyn/tree/main/dyn/datasets/breast_cancer). [cells.txt](https://github.com/wxli0/dyn/blob/main/dyn/datasets/breast_cancer/cells.txt) contains the 2D coordinates of individual cell shapes, separated by newlines. [lines.txt](https://github.com/wxli0/dyn/blob/main/dyn/datasets/breast_cancer/lines.txt) contains the line type for the cells in [cells.txt](https://github.com/wxli0/dyn/blob/main/dyn/datasets/breast_cancer/cells.txt).

To reproduce the results, we have the following notebooks, with results produced using geomstats 2.7.0.

- [osteosarcoma_analysis.ipynb](https://github.com/wxli0/dyn/blob/main%4092c7a58/dyn/notebooks/osteosarcoma_analysis.ipynb): Analysis of the osteosarcoma dataset.

- [osteosarcoma_analysis_no_rescale.ipynb](https://github.com/wxli0/dyn/blob/main%4092c7a58/dyn/notebooks/osteosarcoma_analysis_no_rescale.ipynb): Analysis of the osteosarcoma dataset without rescaling.

- [breast_cancer_analysis.ipynb](https://github.com/wxli0/dyn/blob/main%4092c7a58/dyn/notebooks/breast_cancer_analysis.ipynb): Analysis of the breast cancer dataset.

