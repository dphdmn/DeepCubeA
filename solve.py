import os
import sys
import re
import time
import subprocess
import threading
import numpy as np
import pickle
import environments.n_puzzle


def parse_scramble(scramble_str):
    arr = [int(a) for x in scramble_str.split("/") for a in x.split()]
    return environments.n_puzzle.NPuzzleState(np.array(arr))


def _run_solver(states, batch_size, language, print_fn, progress_callback, output_file=None):
    n = len(states[0].tiles) - 1
    env_name = f"puzzle{n}"
    results_dir = "results"

    here = os.path.dirname(os.path.abspath(__file__))
    data = {"states": states}
    os.makedirs("data", exist_ok=True)
    pickle.dump(data, open(os.path.join(here, "data", "scramble.pkl"), "wb"))

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
        f" --language {language}"
        f" --nnet_batch_size 10000"
    )

    print_fn(f"Running solver (batch_size={batch_size})...")
    start_time = time.time()
    proc = subprocess.Popen(astar_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    def is_noise(line):
        return (not line
                or line.startswith("Times - remOpen:")
                or line.startswith("Iteration:")
                or line.startswith("INITIALIZING")
                or line.startswith("GET START")
                or line.startswith("The argument supplied is")
                or line == "State:"
                or line.startswith("Move nums:")
                or line.startswith("Nodes Generated:")
                or line.startswith("Total time:")
                or line == "SOLVED!"
                or re.match(r'^[\d.\s]+$', line))

    solutions = []

    def stream(stream, label):
        in_progress = False
        for raw in iter(stream.readline, ''):
            line = raw.rstrip()
            itr = None
            m = re.match(r'\s*itr (\d+),', line)
            if m:
                itr = int(m.group(1))
            m = re.match(r'Iteration: (\d+),', line)
            if m:
                itr = int(m.group(1))

            if itr is not None and (itr % 10 == 0 or itr == 1):
                elapsed = time.time() - start_time
                sys.stdout.write(f"\riter {itr:>5d} | {elapsed:>6.1f}s")
                sys.stdout.flush()
                in_progress = True
                if progress_callback:
                    progress_callback(itr)
            elif line.startswith("Solution:"):
                sol = line[len("Solution:"):].strip()
                solutions.append(sol)
                if output_file:
                    with open(output_file, 'a') as f:
                        f.write(sol + '\n')
                if in_progress:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    in_progress = False
                print_fn(line)
            elif is_noise(line):
                pass
            else:
                if in_progress:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    in_progress = False
                print_fn(line)
        if in_progress:
            sys.stdout.write('\n')
            sys.stdout.flush()
        stream.close()

    t1 = threading.Thread(target=stream, args=(proc.stdout, "out"))
    t2 = threading.Thread(target=stream, args=(proc.stderr, "err"))
    t1.start(); t2.start()
    t1.join(); t2.join()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"solver failed with exit code {proc.returncode}")
    print_fn("Done.")
    return solutions


def solve_single(scramble, batch_size, output_file=None, print_fn=print, language="python", progress_callback=None):
    state = parse_scramble(scramble)
    sols = _run_solver([state], batch_size, language, print_fn, progress_callback, output_file)
    return sols[0] if sols else None


def solve_batch(scrambles, batch_size, language, output_file=None, print_fn=print, progress_callback=None):
    states = [parse_scramble(sc) for sc in scrambles]
    return _run_solver(states, batch_size, language, print_fn, progress_callback, output_file)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scramble_or_file", nargs="?", help="Scramble string or file path with --file")
    parser.add_argument("batch_size", nargs="?", type=int, default=4000, help="Batch size (default: 4000)")
    parser.add_argument("--file", action="store_true", help="Read scrambles from file")
    parser.add_argument("--language", choices=["python", "cpp"], default="python", help="Solver backend")
    args = parser.parse_args()

    if args.scramble_or_file is None:
        parser.print_help()
        sys.exit(1)

    if args.file:
        filepath = args.scramble_or_file
        with open(filepath) as f:
            scrambles = [line.strip() for line in f if line.strip()]
        if not scrambles:
            print("No scrambles found.")
            return
        output_file = os.path.abspath(f"{os.path.splitext(filepath)[0]}_solutions.txt")
        print(f"Solving {len(scrambles)} puzzles... -> {output_file}")
        open(output_file, 'w').close()
        solutions = solve_batch(scrambles, args.batch_size, args.language, output_file)
        print(f"\nSolutions ({output_file}):")
        with open(output_file) as f:
            print(f.read(), end='')

    else:
        solve_single(args.scramble_or_file, args.batch_size, language=args.language)


if __name__ == "__main__":
    main()
