# DeepCubeA (sliding puzzles fork)

Sliding Puzzles (such as 15 puzzle, 24 puzzle, 35 puzzle, and 48 puzzle) solver using deep-learning guided A*.  

This fork fixes numpy compatibility issues (only sliding puzzle models included).  
For general information about the project, please visit the [official repo](https://github.com/forestagostinelli/DeepCubeA).

## Quick start (Google Colab)

[Open in Colab](https://colab.research.google.com/github/dphdmn/DeepCubeA/blob/master/deepcubea_colab.ipynb) → set **Runtime → Change runtime type → T4 GPU** → run the cell.

Enter one or more scrambles (one per line) and click **Solve**.  
Solutions append to `solutions.txt` as each puzzle finishes; you can check the file at any time while the solver is running.  
When all scrambles are solved, the file opens automatically in the viewer.

## Scramble format

Space-separated rows separated by `/`. Puzzle size is auto-detected from the number of tiles.

5×5 (n=24) examples:
```
2 10 17 22 18/20 3 8 23 12/21 24 11 14 13/4 7 15 9 6/19 5 0 16 1
22 7 19 8 4/6 11 23 2 5/10 9 13 15 1/16 18 0 12 3/24 17 14 20 21
19 2 0 8 12/6 24 18 23 9/10 22 3 11 15/13 4 17 20 16/1 5 7 14 21
18 10 2 13 19/12 16 7 23 24/20 3 5 17 22/1 6 11 14 21/15 4 0 9 8
16 24 13 19 1/10 8 18 11 0/3 5 9 22 20/23 17 4 14 12/21 7 6 2 15
5 15 24 22 8/13 18 7 16 17/14 10 4 3 19/23 21 12 1 9/2 20 0 11 6
20 7 8 4 15/10 3 19 18 2/0 24 5 21 22/12 11 23 16 9/14 13 6 1 17
21 4 16 10 15/12 6 13 8 22/3 24 20 17 14/23 11 5 9 19/7 2 18 0 1
6 13 24 2 12/9 8 11 21 22/7 19 5 20 18/15 17 4 0 14/1 10 16 3 23
5 9 24 19 14/2 23 17 0 22/8 21 11 4 10/20 1 6 7 3/18 15 12 16 13
```

6×6 (n=35) examples:
```
6 22 18 3 7 23/28 11 9 26 17 24/16 2 25 29 0 33/10 15 4 8 5 34/30 19 31 1 20 27/35 32 13 14 21 12
8 11 1 23 33 24/21 20 30 35 17 26/18 6 29 25 10 28/19 4 7 0 32 5/14 3 15 31 13 2/27 22 16 34 12 9
32 21 6 18 1 28/10 7 13 9 27 24/0 33 5 14 19 8/15 34 4 23 17 25/2 29 12 30 22 26/35 11 16 3 20 31
12 7 25 29 33 11/34 22 32 20 16 8/0 15 1 4 28 2/13 9 19 3 26 30/21 5 31 17 35 18/27 14 24 23 10 6
18 0 5 17 20 35/29 6 27 9 14 3/33 22 7 25 8 34/1 28 23 26 4 30/2 15 24 16 31 12/10 13 21 32 11 19
22 6 19 16 14 26/28 5 23 8 11 18/13 12 9 21 4 32/17 24 29 34 3 25/33 31 20 15 27 35/0 30 7 1 2 10
26 20 15 14 35 16/22 7 30 27 11 25/8 3 1 33 21 17/31 34 2 6 29 23/12 4 18 32 19 13/9 10 24 5 28 0
33 8 27 18 23 2/9 4 5 1 14 29/10 24 25 28 19 32/0 20 15 31 6 17/16 22 11 3 30 21/13 12 26 35 7 34
32 27 25 23 20 17/15 4 33 1 22 21/8 10 12 24 5 6/28 26 16 35 3 14/11 2 9 18 34 31/29 0 30 7 13 19
30 21 8 6 29 31/2 12 34 22 1 4/0 32 18 9 35 23/25 26 15 10 27 17/24 14 11 7 13 3/28 33 20 19 5 16
```

Works for 4×4 (n=15), 5×5 (n=24), 6×6 (n=35), and 7×7 (n=48).

## Batch size

Lower batch size = faster iterations, but may lead to longer (worse) solutions.  
A good batch size is about 4000, but on many scrambles you can get same results even with just 400. Feel free to experiment.

## CLI usage

```bash
# Single scramble
python solve.py "12 6 4 9 18/3 5 17 11 7/13 22 14 24 10/23 2 21 1 20/19 0 16 8 15"

# Batch file (one scramble per line)
python solve.py scrambles.txt --file

# With custom batch size
python solve.py "12 6 4 9 18/3 5 17 11 7/13 22 14 24 10/23 2 21 1 20/19 0 16 8 15" 4000

# Batch + custom batch size + language
python solve.py scrambles.txt --file 4000 --language cpp
```

- `--file` — read scrambles from file, write solutions to `{input}_solutions.txt`
- `batch_size` — positional argument, place after the scramble or before flags (default 4000)
- `--language` — `python` (default) or `cpp`

The file path is printed at the start; solutions are written incrementally as each puzzle is solved so you can grab early results while the rest are still running.

> **Note:** C++ backend (`--language cpp`) requires running `bash compile.sh` first.  
> The compile script is Linux-only and will not work on Windows. Use `--language python` on Windows (default).

## Running locally

Make sure you have all requirements installed.

```
matplotlib>=3.5.0
numpy>=1.24.0
torch>=2.0.0
```

PyTorch with CUDA is strongly recommended for performance.  
It may take a while to install torch properly.  
See [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

## Tested with

- Python 3.12.13 / NumPy 2.0.2 / PyTorch 2.11.0+cu128
- GPU: Tesla T4
