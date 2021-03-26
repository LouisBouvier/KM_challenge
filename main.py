import argparse
import numpy as np
import pandas as pd
import time

from utils import write_csv, run_model
from kernels import Mismatch_kernel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mkl', action='store_true', help="If used, will produce the solution obtained using MKL, otherwise the one with a single mismatch kernel.")
    parser.add_argument('--savefile', type=str, default="test_submission.csv")
    parser.add_argument('--data_folder', type=str, default='data')
    args = parser.parse_args()

    print("Careful ! This script requires computing Mismatch kernels, which can be quite long.")
    # If only single mismatch
    if not args.mkl:
        default_params = {"lamb": 15, "k": 7, "m": 3}

        K = []

        print("Computing kernels...")
        for name in [0, 1, 2]:
            X    = np.array(pd.read_csv(f'{args.data_folder}/Xtr{name}.csv')['seq'])
            X_ev = np.array(pd.read_csv(f'{args.data_folder}/Xte{name}.csv')['seq'])

            t0 = time.time()
            K_tr = Mismatch_kernel(X, X, k=default_params['k'], m=default_params['m'])
            K_te = Mismatch_kernel(X, X_ev, k=default_params['k'], m=default_params['m'])
            print(f"Finished computing mismatch kernel for dataset {name}.")

            K.append({"train": K_tr, "eval": K_te})

        preds, _ = run_model('ksvm', kernel='', K=K, sequence=True, prop_test=0.2, default_params=default_params)

        write_csv(np.arange(preds.shape[0]), preds, args.savefile)

    # If MKL is used
    else:
        default_params = {"lamb": 25, "step": .05}

        kernel_params = [(7, 3), (8, 3)]

        K = [{"train": [], "eval": []} for _ in range(len(kernel_params))]

        print("Computing kernels...")
        for name in [0, 1, 2]:
            X    = np.array(pd.read_csv(f'{args.data_folder}/Xtr{name}.csv')['seq'])
            X_ev = np.array(pd.read_csv(f'{args.data_folder}/Xte{name}.csv')['seq'])

            for (k, m) in kernel_params:

                t0 = time.time()
                K_tr = Mismatch_kernel(X, X, k=k, m=m)
                K_te = Mismatch_kernel(X, X_ev, k=k, m=m)
                print(f"Finished computing mismatch kernel {k}, {m} for dataset {name}.")

                K[name]["train"].append(K_tr)
                K[name]["eval"].append(K_te)

        preds, _ = run_model('ksvm', kernel='', K=K, sequence=True, prop_test=0.2,
                              default_params=default_params, use_mkl=True, mkl_iterations=5)

        write_csv(np.arange(preds.shape[0]), preds, args.savefile)
