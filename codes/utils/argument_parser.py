import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="Argument parser to obtain the name of the config file")
    parser.add_argument('--config_id', default="graph_rel", help='config id to use')
    parser.add_argument('--exp_id', default='', help='optional experiment id to resume')
    parser.add_argument('--dataset', default='', help='specify dataset name, will replace the one in config')
    parser.add_argument('--seed_index', default=9, type=int, help='index of seed, range [0, 9]')
    args = parser.parse_args()
    return args