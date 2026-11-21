
UNSW-NB15 Flow-Based False-Alarm Reduction Baseline
Empirical baseline evaluating benign-only Isolation Forest anomaly scoring + HistGradientBoostingClassifier with threshold calibration and Clopper-Pearson precision bounds on the official UNSW-NB15 flow splits (175,341 train / 82,332 test flows).

Run
Bash
pip install -r requirements.txt
python flow_baseline.py
Verified Output
Plaintext
Dataset: UNSW-NB15 | Train flows: 175341 | Test flows: 82332
Test Benign: 37000 | Test Attack: 45332
ROC-AUC: 0.9849

Threshold  Precision  Recall     Benign FP    Benign FPR   95% CP Low
---------------------------------------------------------------------
0.50       0.8194     0.9844     9833         26.58      % 0.8167    
0.70       0.8908     0.9611     5342         14.44      % 0.8884    
0.85       0.9531     0.9288     2071         5.60       % 0.9514    
0.92       0.9814     0.8875     764          2.06       % 0.9802    

Top 5 Most Influential Features (Shortcut/Leakage Check):
- ackdat: 0.0149
- sttl: 0.0056
- smean: 0.0051
- dinpkt: 0.0041
- sloss: 0.0015
