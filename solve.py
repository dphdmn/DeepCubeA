import os
import sys
import subprocess
import threading
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

    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = f"{here}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"

    astar_cmd = (
        f"python {os.path.join(here, 'search_methods', 'astar.py')}"
        f" --states {os.path.join(here, 'data', 'scramble.pkl')}"
        f" --model_dir {os.path.join(here, 'saved_models', env_name, 'current')}"
        f" --env {env_name}"
        f" --weight 0.8"
        f" --batch_size {batch_size}"
        f" --results_dir {os.path.join(here, results_dir)}"
        f" --language cpp"
        f" --nnet_batch_size 10000"
    )
    if output_file:
        astar_cmd += f" >> \"{output_file}\""

    print_fn(f"Running solver (batch_size={batch_size})...")
    proc = subprocess.Popen(astar_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    def stream(stream, label):
        for line in iter(stream.readline, ''):
            print_fn(line.rstrip())
        stream.close()

    t1 = threading.Thread(target=stream, args=(proc.stdout, "out"))
    t2 = threading.Thread(target=stream, args=(proc.stderr, "err"))
    t1.start(); t2.start()
    t1.join(); t2.join()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"solver failed with exit code {proc.returncode}")
    print_fn("Done.")


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
