import argparse
import pickle

from src.helpers.nice import display_solution


def get_parser():
    parser = argparse.ArgumentParser(description="What is my solution?")
    parser.add_argument("--save_filename", type=str, default="")
    parser.add_argument("--up_to", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    with open(args.save_filename, "rb") as ff:
        output = pickle.load(ff)

    for i in range(args.up_to):
        print(f"Solution {i+1}")
        display_solution(output["population"][i])
