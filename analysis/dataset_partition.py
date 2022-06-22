import os
import pickle
import random
from parameter import Para


def shuffle(data: list):
    random.shuffle(data)
    length = len(data)
    return data[:int(length*0.6)], data[int(length*0.6):int(length*0.8)], data[int(length*0.8):]


if __name__ == "__main__":
    args = Para().get_para()
    with open(os.path.join(args['root_path'], "nidus.pkl"), "rb") as f:
        nidus = pickle.load(f)
    multiple_nidus, huge_nidus, tiny_nidus = [], [], []
    for k in nidus:
        if len(nidus[k]) > 1:
            multiple_nidus.append(k)
        elif nidus[k][0][1] < 50000:
            tiny_nidus.append(k)
        else:
            huge_nidus.append(k)
    multi_train, multi_val, multi_test = shuffle(multiple_nidus)
    tiny_train, tiny_val, tiny_test = shuffle(tiny_nidus)
    huge_train, huge_val, huge_test = shuffle(huge_nidus)
    partition = {"nidus_train": multi_train + tiny_train + huge_train,
                 "nidus_val": multi_val + tiny_val + huge_val,
                 "nidus_test": multi_test + tiny_test + huge_test}
    with open(os.path.join(args['root_path'], "partition.pkl"), "wb") as f:
        pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)