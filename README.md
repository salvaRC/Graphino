# The World as a Graph: Improved El Nino Forecasts with Graph Neural Networks 
[Link to paper](https://arxiv.org/abs/2104.05089) (under review currently) <br>
*Deep learning-based models have recently outperformed state-of-the-art seasonal forecasting models, such as for predicting 
El Ni\~no-Southern Oscillation (ENSO). 
However, current deep learning models are based on convolutional neural networks which are difficult to interpret and can fail to model large-scale atmospheric patterns. In comparison, graph neural networks (GNNs) are capable of modeling large-scale spatial dependencies and are more interpretable due to the explicit modeling of information flow through edge connections.
We propose the first application of graph neural networks to seasonal forecasting.
We design a novel graph connectivity learning module that enables our GNN model to learn large-scale spatial interactions jointly with the actual ENSO forecasting task.
Our model, \graphino, outperforms state-of-the-art deep learning-based
models for forecasts up to six months ahead.
Additionally, we show that our model is more interpretable as it learns sensible connectivity structures that correlate with the ENSO anomaly pattern.*

## Data setup
- Download the datasets from [this link](https://drive.google.com/drive/folders/15L2cvpAQv_c6c6gmJ8RnR2tQ_mHQR9Oz?usp=sharing)
- Place the downloaded data into [this subfolder](Data) (which already has the correct substructure with subdirs SODA, GODAS, CMIP5_CNN).

## Environment setup
Please follow the instructions in [this file](ENVIRONMENT.md).

## Models

All reported models (4 per #lead months) are provided in [the *out* directory](out/).
To reload them & ensemble them as in the paper (and get the reported all season correlation skills), 
you may run the [*eval_gcn*](eval_gcn.py) script for a given number of lead months/horizon.

## Running the experiments

Please run the [*run_graphino*](run_graphino.py) script for the desired number of lead months h in {1,2, .., 23} (the horizon argument).

## Eigenvector centrality analysis and Fig. 2
To produce Figure 2 in our paper, i.e.
a heatmap of eigenvector centrality scores of the nodes for 
various of our GCN models for different lead times, please see [this jupyter notebook](interpretability_plots.ipynb).


## Citation

Please consider citing the following paper if you find it, or the code, helpful. Thank you!

    @article{cachay2021world,
          title={The World as a Graph: Improving El Ni\~no Forecasts with Graph Neural Networks}, 
          author={Salva Rühling Cachay and Emma Erickson and Arthur Fender C. Bucker and Ernest Pokropek and Willa Potosnak and Suyash Bire and Salomey Osei and Björn Lütjens},
          year={2021},
          eprint={2104.05089},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
