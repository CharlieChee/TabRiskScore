import argparse
import os
import pickle
import sys
import warnings

import aiofiles


async def save_metrics_to_file(file_path: str, data):
    """Save metrics to file

    :param file_path: file path
    :type file_path: str
    :param data: data to save
    :type data: _type_
    """
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(pickle.dumps(data))


def str2bool(s: str):
    """convert string to bool. Used in parser.

    :param s: "True" or "False". Raises argparse.ArgumentTypeError if value is neither.
    :type s: str
    :return: boolean
    :rtype: bool
    """
    # This is for boolean type in the parser
    if s == "True":
        return True
    if s == "False":
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(s):
    """Convert a string to a list. Used in parser.

    :param s: string
    :type s: str
    :return: list
    :rtype: list
    """
    # Has to be a list of str
    sub = s[1 : len(s) - 1]
    l = []
    first = True
    tamp = 0
    for i, c in enumerate(sub):
        if c == ",":
            continue
        if c == "'":
            if first:
                tamp = i + 1
                first = False
            else:
                l.append(sub[tamp:i])
                first = True
    return l


# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, "w")
    warnings.filterwarnings("ignore")


# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__
    warnings.filterwarnings("default")


# ignore deprecation warnings
def ignore_depreciation():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
