import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load datasets
df1 = pd.read_csv('5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_attacktype.csv')
df2 = pd.read_csv('5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_final.csv')

print("=" * 80)
print("5G NIDD TIME SERIES DATA ANALYSIS")
print("=" * 80)

# Dataset 1: Multi-class (attack_type)
print("\n### DATASET 1: BTS1_BTS2_timeseries_attacktype.csv (Multi-class)")
print("-" * 80)
print(f"Shape: {df1.shape[0]} samples, {df1.shape[1]} features")
print(f"Time range: {df1['timestamp'].min()} to {df1['timestamp'].max()}")
print(f"Missing values: {df1.isnull().sum().sum()}")

print("\nAttack Type Distribution:")
attack_counts = df1['attack_type'].value_counts().sort_index()
attack_labels = {
    0: 'Benign',
    1: 'HTTPFlood',
    2: 'ICMPFlood',
    3: 'SYNFlood',
    4: 'SYNScan',
    5: 'SlowrateDoS',
    6: 'UDPLagScan',
    7: 'UDPFlood',
    8: 'UDPScan'
}

for attack_type in attack_counts.index:
    count = attack_counts[attack_type]
    percentage = count / len(df1) * 100
    label = attack_labels.get(attack_type, f'Type{attack_type}')
    print(f"  {attack_type} ({label:20s}): {count:5d} samples ({percentage:6.2f}%)")

# Dataset 2: Binary (label)
print("\n### DATASET 2: BTS1_BTS2_timeseries_final.csv (Binary)")
print("-" * 80)
print(f"Shape: {df2.shape[0]} samples, {df2.shape[1]} features")
print(f"Time range: {df2['timestamp'].min()} to {df2['timestamp'].max()}")
print(f"Missing values: {df2.isnull().sum().sum()}")

print("\nLabel Distribution:")
label_counts = df2['label'].value_counts().sort_index()
label_names = {0: 'Normal', 1: 'Malicious'}
for label in label_counts.index:
    count = label_counts[label]
    percentage = count / len(df2) * 100
    name = label_names[label]
    print(f"  {label} ({name:10s}): {count:5d} samples ({percentage:6.2f}%)")

# Feature analysis
print("\n### FEATURE ANALYSIS")
print("-" * 80)
print(f"Total features: {df1.shape[1]}")
print(f"Feature columns (excluding timestamp, label, flow_count):")

feature_cols = [col for col in df1.columns if col not in ['timestamp', 'attack_type', 'label', 'flow_count']]
print(f"  Network flow features: {len(feature_cols)}")

# Key statistics
print("\n### KEY STATISTICS")
print("-" * 80)

key_features = ['TotPkts', 'TotBytes', 'Rate', 'TcpRtt', 'Loss', 'Dur']
print("\nMulti-class dataset - Normal vs Attack comparison:")
for feat in key_features:
    if feat in df1.columns:
        normal_mean = df1[df1['attack_type'] == 0][feat].mean()
        attack_mean = df1[df1['attack_type'] != 0][feat].mean()
        print(f"  {feat:15s}: Normal={normal_mean:12.4f}, Attack={attack_mean:12.4f}")

print("\nBinary dataset - Normal vs Malicious comparison:")
for feat in key_features:
    if feat in df2.columns:
        normal_mean = df2[df2['label'] == 0][feat].mean()
        malicious_mean = df2[df2['label'] == 1][feat].mean()
        print(f"  {feat:15s}: Normal={normal_mean:12.4f}, Malicious={malicious_mean:12.4f}")

# Time series characteristics
print("\n### TIME SERIES CHARACTERISTICS")
print("-" * 80)
print(f"Multi-class dataset:")
print(f"  Flow count range: {df1['flow_count'].min()} to {df1['flow_count'].max()}")
print(f"  Average flow count: {df1['flow_count'].mean():.2f}")
print(f"  Median flow count: {df1['flow_count'].median():.2f}")

print(f"\nBinary dataset:")
print(f"  Flow count range: {df2['flow_count'].min()} to {df2['flow_count'].max()}")
print(f"  Average flow count: {df2['flow_count'].mean():.2f}")
print(f"  Median flow count: {df2['flow_count'].median():.2f}")

# Create visualizations
print("\n### CREATING VISUALIZATIONS...")
print("-" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Attack type distribution
ax1 = fig.add_subplot(gs[0, 0])
attack_counts = df1['attack_type'].value_counts().sort_index()
colors = ['green' if i == 0 else 'red' for i in attack_counts.index]
bars = ax1.bar(range(len(attack_counts)), attack_counts.values, color=colors, alpha=0.7)
ax1.set_xticks(range(len(attack_counts)))
ax1.set_xticklabels([attack_labels.get(i, f'T{i}') for i in attack_counts.index], rotation=45, ha='right')
ax1.set_title('Multi-class: Attack Type Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars, attack_counts.values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(count), 
             ha='center', va='bottom', fontsize=9)

# Plot 2: Binary label distribution
ax2 = fig.add_subplot(gs[0, 1])
label_counts = df2['label'].value_counts().sort_index()
bars = ax2.bar(['Normal', 'Malicious'], label_counts.values, color=['green', 'red'], alpha=0.7)
ax2.set_title('Binary: Label Distribution', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count')
ax2.grid(axis='y', alpha=0.3)
for bar, count in zip(bars, label_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(count), 
             ha='center', va='bottom', fontsize=10)

# Plot 3: Flow count over time
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(df1['timestamp'], df1['flow_count'], alpha=0.6, linewidth=0.8, color='steelblue')
ax3.set_title('Flow Count Over Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Flow Count')
ax3.grid(alpha=0.3)

# Plot 4-6: Feature distributions (TotPkts, TotBytes, Rate)
feature_cols = ['TotPkts', 'TotBytes', 'Rate']
for idx, col in enumerate(feature_cols):
    ax = fig.add_subplot(gs[1, idx])
    if col in df1.columns:
        normal_data = df1[df1['attack_type'] == 0][col].dropna()
        attack_data = df1[df1['attack_type'] != 0][col].dropna()
        bp = ax.boxplot([normal_data, attack_data], labels=['Normal', 'Attack'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)
        ax.set_title(f'{col} Distribution (Multi-class)', fontsize=11, fontweight='bold')
        ax.set_ylabel(col)
        ax.grid(axis='y', alpha=0.3)

# Plot 7-9: More feature distributions
feature_cols2 = ['TcpRtt', 'Loss', 'Dur']
for idx, col in enumerate(feature_cols2):
    ax = fig.add_subplot(gs[2, idx])
    if col in df2.columns:
        normal_data = df2[df2['label'] == 0][col].dropna()
        malicious_data = df2[df2['label'] == 1][col].dropna()
        bp = ax.boxplot([normal_data, malicious_data], labels=['Normal', 'Malicious'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)
        ax.set_title(f'{col} Distribution (Binary)', fontsize=11, fontweight='bold')
        ax.set_ylabel(col)
        ax.grid(axis='y', alpha=0.3)

plt.savefig('time_series_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'time_series_analysis.png'")

# Class imbalance analysis
print("\n### CLASS IMBALANCE ANALYSIS")
print("-" * 80)
print("Multi-class dataset:")
print(f"  Imbalance ratio (Normal:Attack): {attack_counts[0]}/{len(df1) - attack_counts[0]} = {attack_counts[0]/(len(df1) - attack_counts[0]):.2f}:1")
print(f"  Most common attack: {attack_labels[attack_counts[attack_counts.index != 0].idxmax()]} ({attack_counts[attack_counts.index != 0].max()} samples)")
print(f"  Least common attack: {attack_labels[attack_counts[attack_counts.index != 0].idxmin()]} ({attack_counts[attack_counts.index != 0].min()} samples)")

print("\nBinary dataset:")
print(f"  Imbalance ratio (Normal:Malicious): {label_counts[0]}/{label_counts[1]} = {label_counts[0]/label_counts[1]:.2f}:1")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

