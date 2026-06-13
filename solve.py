import os
import sys
import numpy as np
import pickle
import environments.n_puzzle


def main():
    if len(sys.argv) < 2:
        print("Usage: python solve.py <scramble>")
        print("Example: python solve.py \"1 2 3 4 5 6 7 8 0\"")
        sys.exit(1)

    scramble = sys.argv[1]

    arr = [int(a) for x in scramble.split("/") for a in x.split()]
    n = int(len(arr) ** 0.5)
    env_name = f"puzzle{n}"

    state = environments.n_puzzle.NPuzzleState(np.array(arr))
    data = {"states": [state]}
    os.makedirs("data", exist_ok=True)
    pickle.dump(data, open("data/scramble.pkl", "wb"))

    cmd = (
        f"python search_methods/astar.py"
        f" --states data/scramble.pkl"
        f" --model_dir saved_models/{env_name}/current"
        f" --env {env_name}"
        f" --weight 0.8"
        f" --batch_size 10000"
        f" --results_dir results"
        f" --language cpp"
        f" --nnet_batch_size 10000"
    )
    os.system(cmd)


if __name__ == "__main__":
    main()
