import sys
import argparse
import json
import run_experiments


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('function', help='name of the function', type=str)
    parser.add_argument('--rank_em', help='the embedding model rank', type=int)
    parser.add_argument('--model_type', help='the textual information model type', type=str)
    args = parser.parse_args()
    rank = args.rank_em
    model_type = args.model_type
    run_experiments.exp(rank, model_type)



if __name__ == '__main__':
    globals()[sys.argv[1]]()
