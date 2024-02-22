

from main import parse_arguments, seed_everything
import os
import glob
import json
from model_slicing.algorithm.src.data_loader import SQLAttacedLibsvmDataset


def write_json(file_name, data):
    print(f"writting {file_name}...")
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data))


args = parse_arguments()
seed_everything(args.seed)


data_dir = os.path.join(args.data_dir, args.dataset)
train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]


train_loader = SQLAttacedLibsvmDataset(
    train_file,
    args.nfield,
    args.max_filter_col)


write_json(
    f"{args.dataset}_col_cardinalities",
    train_loader.col_cardinalities)

