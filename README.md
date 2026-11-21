# UNSW-NB15 Flow-Based False-Alarm Reduction Baseline

Empirical baseline evaluating benign-only Isolation Forest anomaly scoring + HistGradientBoostingClassifier with threshold calibration and Clopper-Pearson precision bounds on the official UNSW-NB15 flow splits (175,341 train / 82,332 test flows).

## Run

    pip install -r requirements.txt
    python flow_baseline.py

## Verified Output

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




##Dataset Details (UNSW-NB15):
- Source: Created by the Cyber Range Lab at UNSW Canberra (Australian Centre for Cyber Security, Moustafa & Slay, 2015) using the IXIA PerfectStorm traffic generator.
- Official Dataset Page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Exact CSV Splits Used (auto-downloaded in data_loader.py):
  - Training split (UNSW_NB15_training-set.csv): 175,341 flows (56,000 normal, 119,341 attack)
    Mirror: https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv
  - Testing split (UNSW_NB15_testing-set.csv): 82,332 flows (37,000 normal, 45,332 attack)
    Mirror: https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv
- Features & Labels: 42 input flow features (3 categorical: proto, service, state; 39 numerical covering byte/packet rates, TCP timing such as ackdat, TTL values such as sttl, and connection aggregates) after dropping id and attack_cat. Covers normal traffic (label = 0) and 9 attack categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms; label = 1).
