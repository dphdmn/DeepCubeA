import os
import sys
import subprocess
import numpy as np
import pickle
import environments.n_puzzle


def make_scramble_file(scramble_str):
    arr = [int(a) for x in scramble_str.split("/") for a in x.split()]
    n = len(arr) - 1
    state = environments.n_puzzle.NPuzzleState(np.array(arr))
    data = {"states": [state]}
    os.makedirs("data", exist_ok=True)
    pickle.dump(data, open("data/scramble.pkl", "wb"))
    return n


def solve_single(scramble, batch_size, output_file=None, print_fn=print):
    n = make_scramble_file(scramble)
    env_name = f"puzzle{n}"
    results_dir = "results"

    def run(cmd, label):
        print_fn(f"[{label}] Running: {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.stdout:
            print_fn(proc.stdout)
        if proc.stderr:
            print_fn(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"[{label}] failed with exit code {proc.returncode}")
        return proc

    run("export CUDA_VISIBLE_DEVICES=0", "export")
    run(". setup.sh", "setup")
    astar_cmd = (
        f"python search_methods/astar.py"
        f" --states data/scramble.pkl"
        f" --model_dir saved_models/{env_name}/current"
        f" --env {env_name}"
        f" --weight 0.8"
        f" --batch_size {batch_size}"
        f" --results_dir {results_dir}"
        f" --language cpp"
        f" --nnet_batch_size 10000"
    )
    if output_file:
        astar_cmd += f" >> \"{output_file}\""
    run(astar_cmd, "astar")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python solve.py <scramble> [batch_size]")
        print("  python solve.py --file <file.txt> [batch_size]")
        sys.exit(1)

    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Usage: python solve.py --file <file.txt> [batch_size]")
            sys.exit(1)
        filepath = sys.argv[2]
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

        with open(filepath) as f:
            scrambles = [line.strip() for line in f if line.strip()]

        output_file = f"{os.path.splitext(filepath)[0]}_solutions.txt"
        for scramble in scrambles:
            print(f"Solving: {scramble}")
            solve_single(scramble, batch_size, output_file)
    else:
        scramble = sys.argv[1]
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
        solve_single(scramble, batch_size)


if __name__ == "__main__":
    main()
