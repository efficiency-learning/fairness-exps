# FC-POT experiments (FWC protocol)

Reproduces Section 7 of the draft, following Fair Wasserstein Coresets
(Xiong et al., 2024, Sec. 7 and App. C.2).

## Data (place in `data/`)
- `adult-all.csv`: https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv
- `german_data_credit.csv`, `communities_crime.csv`: https://github.com/tailequy/fairness_dataset (experiments/data/)
- `drug_consumption.data`: UCI Drug Consumption (quantified); mirror used: https://github.com/ltempe/Drug-Consumption-Analysis-Project

## Run
    pip install pot scikit-learn pandas scipy matplotlib tabulate
    python fcpot_exp.py --dataset drug   --runs 0-9 --out results
    python fcpot_exp.py --dataset crime  --runs 0-9 --out results
    python fcpot_exp.py --dataset german --runs 0-9 --out results
    python fcpot_exp.py --dataset adult  --runs 0-9 --target_cap 1500 --out results
    python analyze.py results report     # tables.md, summary_table.tex, tradeoff*.png

`run_missing.sh <dataset> <target_cap>` runs only splits whose result file is missing.

Runtime on 2 CPU cores: German/Crime/Drug a few minutes per split; Adult ~20 min per split.

## Notes
- FWC is a reimplementation (official code was not available): majorization-minimization
  with l1 cost; LP (8) solved exactly in a class-aggregated form (checked equal to the
  full LP on test instances); weighted coordinate-wise median updates.
- Fairlets, IndFair coresets and k-median coresets (FWC baselines) are not included yet.
- `results/` holds the raw per-split CSVs used for the paper's table and figures.

## Appendix results
    python gain_quality.py     # approximate-gain ratio vs alpha (results_extra/gain_quality.csv)
    python extra_figs.py       # gain_quality, w1_vs_size figures; per-size, clustering-cost, eps tables

## LLM fairness experiment (extends FWC Sec. 7 / App. C.2)
`results_llm/prompts.jsonl` already holds all 10,000 prompts (Adult + German, 5 seeds,
5 conditions x 200 test points), so only `requests` is needed to query a model:

    export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY=...
    python llm_fairness.py run --provider anthropic --model claude-haiku-4-5-20251001
    python llm_fairness.py run --provider openai --model gpt-4o-mini
    python llm_fairness.py run --provider openai --model <name> --base_url <OpenAI-compatible URL>
    python llm_fairness.py score

Runs resume from `results_llm/responses_<model>.jsonl` if interrupted. Use
`--limit_seeds 1` for a 2,000-call pilot. Rebuild prompts with `python llm_fairness.py prepare`.
