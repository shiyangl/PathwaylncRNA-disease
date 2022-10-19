## pathwayGNN

A pathway-based metapath graph neural network for lncRNA–disease association prediction.

### Dependencies

* python 3.6
* PyTorch 1.2.0
* NetworkX 2.3
* scikit-learn 0.21.3
* NumPy 1.17.2
* SciPy 1.3.1
* DGL 0.3.1



### Datasets

| Entity types | num  | Edge types     | num   |
| ------------ | ---- | -------------- | ----- |
| lncRNA       | 6583 | lncRNA-disease | 8164  |
| miRNA        | 1842 | lncRNA-miRNA   | 2770  |
| Disease      | 1191 | miRNA-disease  | 18732 |
| Total        | 9616 | Total          | 29666 |

### Usage



1.data preprocess: run preprocess_pathwayLDA.ipynb



2.run run_pathwayLDA.py  