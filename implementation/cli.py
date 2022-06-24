import sys
import argparse
import json
import run_experiments
import sys


def check_model_type(model_type):
    list_check = ['LR', 'GNB', 'DT', 'KNN', 'SVM', 'all']
    for x in list_check:
        if model_type == x:
            return
    print("We did not use this type textual information model"), sys.exit()


def check_rank(rank):
    if rank not in [1, 2, 3]:
        print("We did not generate embeddings for the specified rank"), sys.exit()


def run_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('function', help='name of the function', type=str)
    parser.add_argument('--rank_em', help='the embedding model rank', type=int)
    parser.add_argument('--model_type', help='the textual information model type', type=str)
    args = parser.parse_args()
    rank = args.rank_em
    model_type = args.model_type
    check_rank(rank)
    check_model_type(model_type)
    run_experiments.exp(rank, model_type)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
