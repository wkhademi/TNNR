# AI 539 Final Project
Implementation of algorithms presented in "Matrix Completion by Truncated
Nuclear Norm Regularization" by Zhang et al.

To run the code, first install the required packages by running:
```
pip install -r requirements.txt
```

To use the ADMM algorithm on the dataset provided in the paper run:
```
python3 main.py --dataset paper --optimizer admm
```

To use the APGL algorithm on the dataset provided in the paper run:
```
python3 main.py --dataset paper --optimizer apgl 
```

To get the full list of command line options run:
```
python3 main.py -h
```

Reference:
```
D. Zhang, Y. Hu, J. Ye, X. Li and X. He, "Matrix completion by Truncated Nuclear Norm Regularization," 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 2192-2199, doi: 10.1109/CVPR.2012.6247927.
```
