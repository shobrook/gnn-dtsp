import pickle
import os
from utils import create_dataset


def pickle_wrap(filename, callback):
    if os.path.isfile(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
    else:
        output = callback()
        with open(filename, "wb") as new_file:
            pickle.dump(output, new_file)

        return output


i = 0
for input, target in create_dataset(50):
    pickle_wrap("../data/input_{}.pkl".format(i), lambda: input)
    pickle_wrap("../data/target_{}.pkl".format(i), lambda: target)
    i += 1
