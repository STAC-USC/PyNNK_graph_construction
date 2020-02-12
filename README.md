## NNK Graph
Python source code for the paper: [
Graph Construction from Data using Non Negative Kernel regression (NNK Graphs)](https://arxiv.org/abs/1910.09383).

To be presented at [ICASSP 2020](https://2020.ieeeicassp.org/).

## Citing this work
```
@article{shekkizhar2020graph,
    title={Graph Construction from Data by Non-Negative Kernel regression},
    author={Sarath Shekkizhar and Antonio Ortega},
    year={2020},
    booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
}
```
```
@misc{shekkizhar2019graph,
    title={Graph Construction from Data using Non Negative Kernel regression (NNK Graphs)},
    author={Sarath Shekkizhar and Antonio Ortega},
    year={2019},
    eprint={1910.09383},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
### Known Issues
 
- The graph plot does not show edges.
 
This is a shortcoming of the backend being used. 
Only graphs with number of edges ~10,000 or less shows edges in plot. 
The visualization is for demo purposes and this issue will not be fixed in the near future.

- ` numpy.linalg.LinAlgError: {m}-th leading minor of the array is not positive definite`

One or more of the data point is too close to each other. 
Possible fixes: Try changing `sigma` to better distinguish data points or 
increasing `epsilon_high` parameter used in `non_neg_qpsolver` function. 
The issue can also be fixed by increasing command line parameter `thresh`.


