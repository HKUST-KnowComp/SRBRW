
import sys
import gzip

import numpy as np


def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        line = fin.readline()
        # skip dimensional information
        if len(line.strip().split()) == 2:
            line = fin.readline()

        while line:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([float(x) for x in parts[1:]])
                yield word, vals
            line = fin.readline()
