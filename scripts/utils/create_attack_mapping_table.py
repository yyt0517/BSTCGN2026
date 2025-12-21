import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load dataset
df = pd.read_csv('5G_NIDD_TIME_SERIES/BTS1_BTS2_timeseries_attacktype.csv')

# Attack type mapping
attack_data = {
    'Label': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'Attack Type': ['Benign', 'HTTPFlood', 'ICMPFlood', 'SYNFlood', 'SYNScan', 
                    'SlowrateDoS', 'UDPLagScan', 'UDPFlood', 'UDPScan'],
    'Category': ['Normal', 'Flood Attack', 'Flood Attack', 'Flood Attack', 'Scan Attack',
                 'Flood Attack', 'Scan Attack', 'Flood Attack', 'Scan Attack'],
    'Count': [2563, 269, 0, 3, 38, 601, 41, 25, 61],
    'Percentage': [71.17, 7.47, 0.00, 0.08, 1.06, 16.69, 1.14, 0.69, 1.69]
}

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('5G NIDD Attack Type Mapping - Complete Reference', 
             fontsize=18, fontweight='bold', y=0.98)

# Table 1: Attack Type Mapping
ax1.axis('tight')
ax1.axis('off')

table_data = []
colors = []

for i in range(len(attack_data['Label'])):
    label = attack_data['Label'][i]
    attack_type = attack_data['Attack Type'][i]
    category = attack_data['Category'][i]
    count = attack_data['Count'][i]
    percentage = attack_data['Percentage'][i]
    
    # Calculate imbalance ratio
    if count > 0 and label != 0:
        imbalance = f"1:{2563/count:.1f}"
    elif label == 0:
        imbalance = "Baseline"
    else:
        imbalance = "N/A"
    
    table_data.append([
        str(label),
        attack_type,
        category,
        f"{count:,}",
        f"{percentage:.2f}%",
        imbalance
    ])
    
    # Color coding
    if label == 0:
        colors.append(['#90EE90'] * 6)  # Light green for benign
    elif count == 0:
        colors.append(['#D3D3D3'] * 6)  # Gray for missing
    elif count < 50:
        colors.append(['#FFB6C6'] * 6)  # Light red for very rare
    elif category == 'Flood Attack':
        colors.append(['#FFD4D4'] * 6)  # Light red for flood
    else:
        colors.append(['#FFE4B5'] * 6)  # Light orange for scan

# Add total row
table_data.append(['TOTAL', 'All Classes', '-', '3,601', '100.00%', '-'])
colors.append(['#E0E0E0'] * 6)

table1 = ax1.table(cellText=table_data,
                   colLabels=['Label', 'Attack Type', 'Category', 'Count', 'Percentage', 'Imbalance Ratio'],
                   cellLoc='center',
                   loc='center',
                   cellColours=colors,
                   colWidths=[0.08, 0.18, 0.18, 0.12, 0.12, 0.15])

table1.auto_set_font_size(False)
table1.set_fontsize(11)
table1.scale(1, 2.5)

# Style header
for i in range(6):
    cell = table1[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Style total row
for i in range(6):
    cell = table1[(len(table_data), i)]
    cell.set_text_props(weight='bold', fontsize=11)

ax1.set_title('Attack Type → Integer Label Mapping', 
              fontsize=14, fontweight='bold', pad=20)

# Table 2: Key Statistics by Attack Type
ax2.axis('tight')
ax2.axis('off')

stats_data = []
stats_colors = []

# Get statistics for each attack type
for label in sorted(df['attack_type'].unique()):
    attack_type = attack_data['Attack Type'][label]
    subset = df[df['attack_type'] == label]
    
    if len(subset) > 0:
        avg_pkts = subset['TotPkts'].mean()
        avg_bytes = subset['TotBytes'].mean()
        avg_rtt = subset['TcpRtt'].mean()
        avg_dur = subset['Dur'].mean()
        avg_loss = subset['Loss'].mean()
        
        stats_data.append([
            str(label),
            attack_type,
            f"{avg_pkts:.1f}",
            f"{avg_bytes:.0f}",
            f"{avg_rtt:.4f}",
            f"{avg_dur:.2f}",
            f"{avg_loss:.3f}"
        ])
        
        # Color coding same as above
        if label == 0:
            stats_colors.append(['#90EE90'] * 7)
        elif len(subset) < 50:
            stats_colors.append(['#FFB6C6'] * 7)
        elif attack_data['Category'][label] == 'Flood Attack':
            stats_colors.append(['#FFD4D4'] * 7)
        else:
            stats_colors.append(['#FFE4B5'] * 7)

table2 = ax2.table(cellText=stats_data,
                   colLabels=['Label', 'Attack Type', 'Avg Pkts', 'Avg Bytes', 'Avg RTT', 'Avg Dur', 'Avg Loss'],
                   cellLoc='center',
                   loc='center',
                   cellColours=stats_colors,
                   colWidths=[0.08, 0.18, 0.12, 0.14, 0.12, 0.12, 0.12])

table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1, 2.5)

# Style header
for i in range(7):
    cell = table2[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

ax2.set_title('Key Feature Statistics by Attack Type', 
              fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='Benign (Normal Traffic)'),
    mpatches.Patch(facecolor='#FFD4D4', edgecolor='black', label='Flood Attacks'),
    mpatches.Patch(facecolor='#FFE4B5', edgecolor='black', label='Scan Attacks'),
    mpatches.Patch(facecolor='#FFB6C6', edgecolor='black', label='Rare Classes (<50 samples)'),
    mpatches.Patch(facecolor='#D3D3D3', edgecolor='black', label='Missing Classes (0 samples)')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
          fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

# Add notes
notes_text = """
Notes:
• Total Samples: 3,601 (1 hour, 1-second intervals)
• Classes: 9 (0-8, class 2 missing)
• Imbalance Ratio: Benign:Attack = 2,563:1,038 = 2.47:1
• Most Common Attack: SlowrateDoS (601 samples, 16.69%)
• Rarest Attack: SYNFlood (3 samples, 0.08%)
• Missing: ICMPFlood (0 samples)

Key Observations:
• Flood attacks have higher TCP RTT and longer duration
• Scan attacks have shorter duration and lower packet counts
• Attack traffic generally has fewer packets and bytes than benign traffic
"""

fig.text(0.5, 0.02, notes_text, ha='center', va='top', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.12, 1, 0.96])
plt.savefig('attack_type_mapping_table.png', dpi=150, bbox_inches='tight')
print("✓ Attack type mapping table saved as 'attack_type_mapping_table.png'")

# Also create a simple CSV for reference
mapping_df = pd.DataFrame(attack_data)
mapping_df['Imbalance_Ratio'] = mapping_df.apply(
    lambda row: f"1:{2563/row['Count']:.2f}" if row['Count'] > 0 and row['Label'] != 0 
    else ("Baseline" if row['Label'] == 0 else "N/A"), axis=1
)
mapping_df.to_csv('attack_type_mapping.csv', index=False)
print("✓ Attack type mapping CSV saved as 'attack_type_mapping.csv'")

print("\n" + "="*80)
print("ATTACK TYPE MAPPING TABLE CREATED")
print("="*80)

