#!/bin/sh
# Run only the (dataset, run) pairs whose result file is missing.
E=/home/claude/fcpot/experiments
DS="$1"; shift; CAP="$1"
for r in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $E/results/${DS}_run$r.csv ] && continue
  python3 $E/fcpot_exp.py --dataset $DS --runs $r --target_cap $CAP --out $E/results
done
