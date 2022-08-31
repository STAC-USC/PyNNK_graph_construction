## NNK Graph
Python source code for the paper: [
Graph Construction from Data using Non Negative Kernel regression (NNK Graphs)](https://arxiv.org/abs/1910.09383).


### Demo
Check out our notebook demo in Google colab [nnk_cpu_demo.ipynb](https://colab.research.google.com/drive/1jwp9N9eRzquaEjC00AdYHcl5tbilDuC2?usp=sharing)
 that shows how to use NNK neighborhoods for classification in toy datasets.


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

-----
Update: Nov, 2021

## Approximate NNK neighbors 
- Solves a batched iterative version of NNK optimization for data points using 
[FAISS](https://github.com/facebookresearch/faiss) and [PyTorch](https://pytorch.org/)
  - Tested with `faiss-gpu==1.7.1.post2`, `torch==1.9.0`
- `approximate_nnk_test.py` provides an example of how the code can be used with torch feature vectors 
using a normalized cosine kernel (range in [0,1]) similarity metric. 
