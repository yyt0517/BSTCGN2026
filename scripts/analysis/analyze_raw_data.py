import pandas as pd
import numpy as np
import os

print("="*100)
print("5G-NIDD Raw Data Analysis".center(100))
print("="*100)

# Attack type mapping based on the image provided
attack_mapping = {
    'Benign': 0,
    'Goldeneye': 1,      # HTTPFlood
    'ICMPFlood': 2,      # ICMPFlood
    'SYNFlood': 3,       # SYNFlood
    'SYNScan': 4,        # SYNScan
    'Slowloris': 5,      # SlowrateDoS
    'TCPConnect': 6,     # UDPLagScan (?)
    'Torshammer': 7,     # UDPFlood (?)
    'UDPFlood': 7,       # UDPFlood
    'UDPScan': 8,        # UDPScan
    'SSH': 4,            # Might be SYNScan
}

# List all attack files
bs1_files = [
    'Goldeneye1.csv',
    'ICMPFlood1.csv',
    'SYNFlood1.csv',
    'SYNScan1.csv',
    'Slowloris1.csv',
    'TCPConnect1.csv',
    'Torshammer1.csv',
    'UDPFlood1.csv',
    'UDPScan1.csv',
    'SSH1.csv'
]

bs2_files = [
    'Goldeneye2.csv',
    'ICMPFlood2.csv',
    'SYNFlood2.csv',
    'SYNScan2.csv',
    'Slowloris2.csv',
    'TCPConnect2.csv',
    'Torshammer2.csv',
    'UDPFlood2.csv',
    'UDPScan2.csv',
    'SSH2.csv'
]

print("\n" + "="*100)
print("BS1 Attack Files Analysis".center(100))
print("="*100)

for filename in bs1_files:
    filepath = f'5G-NIDD-raw/BS1_each_attack_csv/{filename}'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        attack_name = filename.replace('1.csv', '')
        
        print(f"\n{attack_name}:")
        print(f"  File: {filename}")
        print(f"  Samples: {len(df):,}")
        print(f"  StartTime range: {df['StartTime'].min()} to {df['StartTime'].max()}")
        
        if 'Label' in df.columns:
            print(f"  Label values: {df['Label'].unique()}")
        if 'Attack Type' in df.columns:
            print(f"  Attack Type values: {df['Attack Type'].unique()}")
        
        # Check time format
        print(f"  First 3 StartTime: {df['StartTime'].head(3).tolist()}")
        print(f"  Last 3 StartTime: {df['StartTime'].tail(3).tolist()}")

print("\n" + "="*100)
print("BS2 Attack Files Analysis".center(100))
print("="*100)

for filename in bs2_files:
    filepath = f'5G-NIDD-raw/BS2_each_attack_csv/{filename}'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        attack_name = filename.replace('2.csv', '')
        
        print(f"\n{attack_name}:")
        print(f"  File: {filename}")
        print(f"  Samples: {len(df):,}")
        print(f"  StartTime range: {df['StartTime'].min()} to {df['StartTime'].max()}")
        
        if 'Label' in df.columns:
            print(f"  Label values: {df['Label'].unique()}")
        if 'Attack Type' in df.columns:
            print(f"  Attack Type values: {df['Attack Type'].unique()}")

print("\n" + "="*100)
print("Checking Combined CSV Files".center(100))
print("="*100)

# Check BTS_1.csv and BTS_2.csv
for filename in ['BTS_1.csv', 'BTS_2.csv', 'Combined.csv']:
    filepath = f'5G-NIDD-raw/{filename}'
    if os.path.exists(filepath):
        print(f"\n{filename}:")
        df = pd.read_csv(filepath, nrows=5)
        print(f"  Columns: {df.columns.tolist()[:15]}")
        print(f"  Shape: {df.shape}")
        
        # Read full file to check attack types
        df_full = pd.read_csv(filepath)
        if 'Attack Type' in df_full.columns:
            print(f"  Attack Type distribution:")
            print(df_full['Attack Type'].value_counts())
        if 'Label' in df_full.columns:
            print(f"  Label distribution:")
            print(df_full['Label'].value_counts())

print("\n" + "="*100)
print("Analysis Complete".center(100))
print("="*100)

