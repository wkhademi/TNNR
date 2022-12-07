# AI 539 Final Project
Implementation of algorithms presented in "Matrix Completion by Truncated
Nuclear Norm Regularization" by Zhang et al.

To run the code, first install the required packages by running:
```
pip install -r requirements.txt
```

To use the ADMM algorithm with synthetic data (sigma=1) run:
```
python3 main.py --data_root data/ --dataset synthetic --img_size 100 200 --r 15 --sigma 1 --p 0.7 --optimizer admm --min_rank 15 --max_rank 15 --alg_max_itrs 30 --opt_max_itrs 200 --opt_tol 1e-8 --rho 1
```

To use the APGL algorithm with synthetic data (sigma=1) run:
```
python3 main.py --data_root data/ --dataset synthetic --img_size 100 200 --r 15 --sigma 1 --p 0.7 --optimizer apgl --min_rank 15 --max_rank 15 --alg_max_itrs 50 --opt_max_itrs 25 --lam 0.04
```


To use the ADMM algorithm on the dataset provided in the paper (e.g, for example image \#1) run:
```
python3 main.py --data_root data/ --dataset real --img_num 1 --corruption text --sigma 0 --optimizer admm --min_rank 6 --max_rank 6 --alg_max_itrs 25 --opt_max_itrs 200 --rho 1
```

To use the APGL algorithm on the dataset provided in the paper (e.g, for example image \#1) run:
```
python3 main.py --data_root data/ --dataset real --img_num 4 --corruption text --sigma 0 --optimizer apgl --min_rank 20 --max_rank 20 --alg_max_itrs 35 --opt_max_itrs 30 --lam 0.7
```

To get the full list of command line options run:
```
python3 main.py -h
```

Reference:
```
D. Zhang, Y. Hu, J. Ye, X. Li and X. He, "Matrix completion by Truncated Nuclear Norm Regularization," 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 2192-2199, doi: 10.1109/CVPR.2012.6247927.
```
