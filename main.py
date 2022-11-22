import os
import argparse


def run():
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type='str', required=True,
        help='Dataset used for solving matrix completion problem')
    parser.add_argument('--corruption', type='str', default='text',
        help='Corruption method for corrupting image. Options are: [text | noise | block]')
    parser.add_argument('--rate', type=float, default=0.25,
        help='Amount of image to corrupt. Only used for noise or block corruption.')
    parser.add_argument('--optimizer', type='str', required=True,
        help='Optimizer used to solve the matrix completion problem. Options are: [admm | apgl]')
    parser.add_argument('--tol', type=float, default=1e-5)
