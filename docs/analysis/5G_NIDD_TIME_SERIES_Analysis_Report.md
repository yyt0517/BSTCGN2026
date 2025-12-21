# 5G NIDD Time Series Data Analysis Report

## Overview

This report analyzes two time series datasets from the 5G NIDD (Non-IP Data Delivery) network traffic:
1. **BTS1_BTS2_timeseries_attacktype.csv** - Multi-class classification dataset
2. **BTS1_BTS2_timeseries_final.csv** - Binary classification dataset

---

## Dataset 1: Multi-class Classification (Attack Type)

### Basic Information
- **File**: `5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_attacktype.csv`
- **Samples**: 3,601 time steps
- **Features**: 86 columns
- **Time Range**: 0 to 3,600 (1 hour with 1-second intervals)
- **Missing Values**: 0
- **Target Variable**: `attack_type` (9 classes: 0-8, with class 2 missing)

### Class Distribution

| Attack Type | Label | Count | Percentage |
|------------|-------|-------|------------|
| 0 | Benign | 2,563 | 71.17% |
| 1 | HTTPFlood | 269 | 7.47% |
| 2 | ICMPFlood | 0 | 0.00% |
| 3 | SYNFlood | 3 | 0.08% |
| 4 | SYNScan | 38 | 1.06% |
| 5 | SlowrateDoS | 601 | 16.69% |
| 6 | UDPLagScan | 41 | 1.14% |
| 7 | UDPFlood | 25 | 0.69% |
| 8 | UDPScan | 61 | 1.69% |

### Class Imbalance Analysis
- **Benign vs Attack Ratio**: 2,563:1,038 = **2.47:1**
- **Most Common Attack**: SlowrateDoS (601 samples, 16.69%)
- **Least Common Attack**: SYNFlood (3 samples, 0.08%)
- **Missing Class**: ICMPFlood (class 2) has 0 samples in this dataset
- **Severe Imbalance**: Classes 3, 4, 6, 7 have very few samples (<50)

---

## Dataset 2: Binary Classification (Normal/Malicious)

### Basic Information
- **File**: `5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_final.csv`
- **Samples**: 3,601 time steps
- **Features**: 86 columns
- **Time Range**: 0 to 3,600 (1 hour with 1-second intervals)
- **Missing Values**: 0
- **Target Variable**: `label` (2 classes: 0=Normal, 1=Malicious)

### Class Distribution

| Label | Type | Count | Percentage |
|-------|------|-------|------------|
| 0 | Normal | 2,532 | 70.31% |
| 1 | Malicious | 1,069 | 29.69% |

### Class Imbalance Analysis
- **Normal vs Malicious Ratio**: 2,532:1,069 = **2.37:1**
- **More Balanced**: Compared to multi-class, binary classification is more balanced
- **Note**: Binary dataset has 31 fewer normal samples than multi-class (2,532 vs 2,563)

---

## Feature Analysis

### Feature Categories (83 network flow features)

The datasets contain comprehensive network flow statistics:

1. **Temporal Features**: timestamp, Dur, RunTime, Mean, StdDev, Sum, Min, Max
2. **Packet Statistics**: TotPkts, SrcPkts, DstPkts, sMeanPktSz, dMeanPktSz
3. **Byte Statistics**: TotBytes, SrcBytes, DstBytes, TotAppByte, SAppBytes, DAppBytes
4. **Network Layer**: SrcMac, DstMac, SrcOui, DstOui, sTos, dTos, sTtl, dTtl, sHops, dHops
5. **Transport Layer**: SrcWin, DstWin, SrcTCPBase, DstTCPBase, TcpRtt, SynAck, AckDat
6. **Flow Metrics**: Rate, SrcRate, DstRate, Load, SrcLoad, DstLoad
7. **Quality Metrics**: Loss, SrcLoss, DstLoss, pLoss, Retrans, SrcRetra, DstRetra, pRetran
8. **Inter-arrival Times**: SIntPkt, SIntDist, SIntPktAct, SIntActDist, DIntPkt, DIntDist
9. **Jitter**: SrcJitAct, DstJitter, DstJitAct
10. **Other**: Rank, Trans, Seq, Offset, PCRatio, flow_count

### Key Feature Statistics Comparison

#### Multi-class Dataset (Normal vs Attack)

| Feature | Normal (Mean) | Attack (Mean) | Difference |
|---------|---------------|---------------|------------|
| TotPkts | 71.72 | 24.24 | **-66.2%** |
| TotBytes | 84,988.87 | 22,547.93 | **-73.5%** |
| Rate | 897.85 | 1,001.25 | **+11.5%** |
| TcpRtt | 0.0027 | 0.0144 | **+433%** |
| Loss | 0.1978 | 0.0554 | **-72.0%** |
| Dur | 2.51 | 3.03 | **+20.7%** |

**Key Observations**:
- Attack traffic has **significantly fewer packets and bytes**
- Attack traffic has **much higher TCP RTT** (network delay)
- Attack traffic has **lower packet loss** but **longer duration**
- Attack traffic has **slightly higher rate**

#### Binary Dataset (Normal vs Malicious)

| Feature | Normal (Mean) | Malicious (Mean) | Difference |
|---------|---------------|------------------|------------|
| TotPkts | 72.53 | 23.71 | **-67.3%** |
| TotBytes | 85,970.63 | 22,033.28 | **-74.4%** |
| Rate | 908.82 | 972.29 | **+7.0%** |
| TcpRtt | 0.0027 | 0.0140 | **+418%** |
| Loss | 0.2002 | 0.0538 | **-73.1%** |
| Dur | 2.52 | 3.00 | **+19.2%** |

**Key Observations**:
- Similar patterns to multi-class dataset
- Malicious traffic characteristics align with attack traffic patterns

---

## Time Series Characteristics

### Flow Count Analysis

Both datasets share identical flow count patterns:
- **Range**: 0 to 57,794 flows per time step
- **Mean**: 686.88 flows
- **Median**: 57.00 flows
- **High Variance**: Large difference between mean and median indicates skewed distribution

### Temporal Patterns
- **Sampling Rate**: 1 second intervals
- **Duration**: 3,600 seconds (1 hour)
- **Continuous**: No gaps in timestamps (0, 1, 2, ..., 3600)

---

## Data Quality Assessment

### Strengths
✓ **No Missing Values**: Complete dataset with no null values
✓ **Consistent Structure**: Both datasets have identical feature sets
✓ **Temporal Continuity**: No gaps in time series
✓ **Rich Features**: 83 network flow features provide comprehensive information
✓ **Real-world Data**: From actual 5G network traffic

### Challenges
⚠ **Class Imbalance**: 
  - Multi-class: Severe imbalance (71% normal, some attacks <1%)
  - Binary: Moderate imbalance (70% normal, 30% malicious)

⚠ **Rare Attack Types**: 
  - Reconnaissance: only 3 samples (0.08%)
  - Malware: only 25 samples (0.69%)
  - Theft: only 38 samples (1.06%)

⚠ **Feature Scale Variance**: 
  - Some features range from 0-1 (e.g., TcpRtt)
  - Others range to millions (e.g., TotBytes, SrcTCPBase)
  - **Normalization required** for most ML models

⚠ **High Dimensionality**: 
  - 83 features may lead to curse of dimensionality
  - Feature selection/reduction may be beneficial

---

## Attack Type Descriptions

Based on the label mapping:

- **Class 0 (Benign)**: Normal, legitimate network traffic
- **Class 1 (HTTPFlood)**: HTTP flood DDoS attack - overwhelming server with HTTP requests
- **Class 2 (ICMPFlood)**: ICMP flood attack - not present in this dataset
- **Class 3 (SYNFlood)**: TCP SYN flood attack - exhausting server resources with half-open connections
- **Class 4 (SYNScan)**: TCP SYN port scanning - reconnaissance activity
- **Class 5 (SlowrateDoS)**: Slow-rate DoS attack - low-and-slow attack to evade detection
- **Class 6 (UDPLagScan)**: UDP lag scanning - network reconnaissance
- **Class 7 (UDPFlood)**: UDP flood attack - overwhelming target with UDP packets
- **Class 8 (UDPScan)**: UDP port scanning - reconnaissance activity

### Attack Categories

The attacks can be grouped into:
1. **Flooding Attacks** (1, 3, 5, 7): HTTPFlood, SYNFlood, SlowrateDoS, UDPFlood
2. **Scanning Attacks** (4, 6, 8): SYNScan, UDPLagScan, UDPScan

---

## Recommendations for Machine Learning

### 1. Data Preprocessing

**For Multi-class Classification**:
- Consider **merging rare classes** (3, 4, 6, 7) into "Other Attacks" or group by attack type (Flood vs Scan)
- Apply **SMOTE or ADASYN** for minority class oversampling
- Use **class weights** in loss function
- Consider **focal loss** for handling extreme imbalance
- **Remove class 2 (ICMPFlood)** as it has no samples

**For Binary Classification**:
- Less severe imbalance, but still consider **class weights**
- May not need aggressive oversampling

**General**:
- **Normalize/Standardize** features (StandardScaler or MinMaxScaler)
- Check for **multicollinearity** (many features are derived from others)
- Consider **feature selection** (e.g., mutual information, feature importance)
- Handle **zero-variance features** (e.g., Trans, StdDev, SrcMac, DstMac all = 0)

### 2. Model Selection

**Recommended Models**:
- **Tree-based**: Random Forest, XGBoost, LightGBM (handle imbalance well)
- **Neural Networks**: LSTM, GRU, Transformer (for temporal patterns)
- **Graph Neural Networks**: GAT, GCN (if flow relationships can be modeled)
- **Ensemble Methods**: Combine multiple models for robustness

**For Time Series**:
- Consider **sliding window** approach for temporal context
- Use **sequence models** (LSTM/GRU) to capture temporal dependencies
- Explore **attention mechanisms** for important time steps

### 3. Evaluation Metrics

**Do NOT use accuracy** (due to class imbalance)

**Use instead**:
- **Precision, Recall, F1-Score** (per class)
- **Macro/Weighted F1-Score** (overall performance)
- **Confusion Matrix** (detailed error analysis)
- **ROC-AUC** (binary) or **Macro/Micro AUC** (multi-class)
- **PR-AUC** (Precision-Recall curve, better for imbalanced data)

### 4. Cross-Validation Strategy

- **Time Series Split**: Respect temporal order (no future leakage)
- **Stratified K-Fold**: Maintain class distribution in each fold
- **Leave-One-Out** for rare classes (if computationally feasible)

### 5. Feature Engineering

**Potential New Features**:
- **Ratio features**: e.g., SrcPkts/DstPkts, SrcBytes/DstBytes
- **Rate features**: e.g., Bytes/Duration, Pkts/Duration
- **Temporal aggregations**: rolling mean, std, min, max over windows
- **Lag features**: previous time step values
- **Change features**: difference from previous time step
- **Attack-specific features**:
  - For flood attacks: high packet rate, low diversity
  - For scan attacks: many unique destinations, small packet sizes

---

## Comparison: Multi-class vs Binary

| Aspect | Multi-class | Binary |
|--------|-------------|--------|
| **Complexity** | Higher (9 classes) | Lower (2 classes) |
| **Imbalance** | Severe (some <1%) | Moderate (30/70) |
| **Use Case** | Detailed attack identification | Quick threat detection |
| **Model Difficulty** | Harder to train | Easier to train |
| **Interpretability** | More informative | Less detailed |
| **Practical Value** | Security analysis | Real-time filtering |

**Recommendation**: 
- Start with **binary classification** for baseline performance
- Progress to **multi-class** for detailed attack type identification
- Consider **hierarchical approach**: binary first, then multi-class on detected attacks

---

## Visualization Summary

The generated visualization (`time_series_analysis.png`) includes:
1. **Attack Type Distribution** (multi-class)
2. **Label Distribution** (binary)
3. **Flow Count Over Time** (temporal pattern)
4. **TotPkts, TotBytes, Rate** distributions (multi-class)
5. **TcpRtt, Loss, Dur** distributions (binary)

---

## Conclusion

These datasets provide rich time series data for 5G network intrusion detection:

**Strengths**:
- Comprehensive network flow features
- Real-world 5G traffic data
- Both binary and multi-class labels available
- Clean data with no missing values

**Challenges**:
- Significant class imbalance (especially multi-class)
- High dimensionality requiring feature engineering
- Rare attack types may be difficult to detect

**Next Steps**:
1. Implement preprocessing pipeline (normalization, feature selection)
2. Build baseline models (Random Forest, XGBoost)
3. Experiment with time series models (LSTM, GRU)
4. Apply class balancing techniques
5. Evaluate with appropriate metrics (F1, AUC)
6. Consider graph-based approaches if flow relationships exist

---

*Analysis generated on: 2025-11-10*
*Datasets: 5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_*.csv*

