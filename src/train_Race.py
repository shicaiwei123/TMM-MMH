"""
Function: Training ResNet with CASIA-Race for WACV2021
Author: AJ
Date: 2021.1.30
"""
import sys

sys.path.append('..')
import argparse, sys
from datasets.cefa_dataset_class import *


def main(args):
    ### Load data ###
    data_train = load_casia_race(args.path_data_root, protocol=args.protocol, mode=args.phases[0])
    image_list, label_list = get_sframe_paths_labels(data_train, args.phases[0], ratio=1)
    print(image_list[1:10],label_list[1:10])
    print('Load Train Data: num_images={}/num_ID={}'.format(len(image_list), len(label_list)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data_root", type=str,
                        default='/home/data/shicaiwei/cefa/CeFA-Race')

    parser.add_argument("--protocol", type=str, default='race_prot_rdi_4@5')
    parser.add_argument("--phases", type=list, default=['train', 'dev', 'test'])
    parser.add_argument("--seed", type=int, default=6)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
