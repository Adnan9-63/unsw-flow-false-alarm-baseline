# UNSW-NB15 Flow-Based False-Alarm Reduction Baseline

Empirical baseline evaluating benign-only Isolation Forest anomaly scoring + HistGradientBoostingClassifier with threshold calibration and Clopper-Pearson precision bounds on the official UNSW-NB15 flow splits (`175,341` train / `82,332` test flows).

## Run

```bash
pip install -r requirements.txt
python flow_baseline.py
