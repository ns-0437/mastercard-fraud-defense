# Raw datasets (not committed)

Download these manually (Kaggle CLI or web UI) into this folder — they're gitignored,
too large and license-scoped to commit.

1. **PaySim** — https://www.kaggle.com/datasets/ealaxi/paysim1
   `kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip`
   Expected file: `data/raw/PS_20174392719_1491204439457_log.csv`

2. **Credit Card Fraud Detection (ULB)** — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip`
   Expected file: `data/raw/creditcard.csv`

Requires a Kaggle API token (`~/.kaggle/kaggle.json`) if using the CLI.
