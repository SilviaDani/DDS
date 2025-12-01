import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_corruption_data(json_file):
    """Load corruption evaluation results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics_for_plots(data):
    """
    Adapted to extract metrics from the 'corruptions' -> 'type' -> 'severity' structure.
    """
    all_data = []
    
    if 'corruptions' not in data:
        print("Error: JSON does not contain 'corruptions' key.")
        return pd.DataFrame()
    
    for corruption, severity_data in data['corruptions'].items():
        for severity_str, metrics in severity_data.items():
            try:
                severity = int(severity_str)
            except ValueError:
                continue 
            
            if not metrics or 'mAP' not in metrics:
                continue
                
            entry = {
                'severity': severity,
                'corruption': corruption,
                'mAP': metrics['mAP']
            }
            
            if 'mean_dds' in metrics:
                entry['dds'] = metrics['mean_dds']
            elif 'dds' in metrics:
                entry['dds'] = metrics['dds']

            if 'mean_lpips' in metrics:
                entry['lpips'] = metrics['mean_lpips']
            elif 'lpips' in metrics:
                entry['lpips'] = metrics['lpips']
            
            all_data.append(entry)
    
    return pd.DataFrame(all_data)

def compute_correlations_by_severity(df):
    """Compute Pearson correlations for each severity level"""
    correlations = {}
    
    for severity in sorted(df['severity'].unique()):
        severity_df = df[df['severity'] == severity].dropna()
        
        if len(severity_df) > 1:
            dds_corr = np.nan
            if 'dds' in severity_df.columns and severity_df['dds'].std() > 0:
                dds_corr, _ = pearsonr(severity_df['dds'], severity_df['mAP'])
            
            lpips_corr = np.nan
            if 'lpips' in severity_df.columns and severity_df['lpips'].std() > 0:
                lpips_corr, _ = pearsonr(severity_df['lpips'], severity_df['mAP'])
            
            correlations[severity] = {
                'dds_map_correlation': dds_corr,
                'lpips_map_correlation': lpips_corr,
                'n_samples': len(severity_df)
            }
    
    return correlations

def create_correlation_plots(correlations, output_dir):
    """Plot correlation coefficients vs Severity (No Stars, Adjusted Axis)"""
    severities = sorted(correlations.keys())
    
    # Filter valid data
    valid_severities = [s for s in severities if not np.isnan(correlations[s]['dds_map_correlation'])]
    
    if not valid_severities:
        print("Not enough data to create correlation plots.")
        return

    dds_corrs = [correlations[s]['dds_map_correlation'] for s in valid_severities]
    lpips_corrs = [correlations[s]['lpips_map_correlation'] for s in valid_severities]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PLOT 1: DDS ---
    ax1.plot(valid_severities, dds_corrs, 'o-', linewidth=3, markersize=10, 
             color='#2E86AB', markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Severity Level', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pearson Correlation (DDS vs MAP)', fontsize=14, fontweight='bold') # Capitalized MAP
    ax1.set_title('DDS Correlation with MAP', fontsize=16, fontweight='bold')
    ax1.set_xticks(valid_severities)
    ax1.grid(True, alpha=0.3)
    
    # Range -1 to 0
    ax1.set_ylim(-1.05, 0.05) 
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Annotate DDS (NO STARS)
    for s, corr in zip(valid_severities, dds_corrs):
        ax1.annotate(f'{corr:.2f}', (s, corr), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=11)

    # --- PLOT 2: LPIPS ---
    ax2.plot(valid_severities, lpips_corrs, 's-', linewidth=3, markersize=10,
             color='#A23B72', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Severity Level', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Pearson Correlation (LPIPS vs MAP)', fontsize=14, fontweight='bold') # Capitalized MAP
    ax2.set_title('LPIPS Correlation with MAP', fontsize=16, fontweight='bold')
    ax2.set_xticks(valid_severities)
    ax2.grid(True, alpha=0.3)

    # Range -1 to 0
    ax2.set_ylim(-1.05, 0.05)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Annotate LPIPS (NO STARS)
    for s, corr in zip(valid_severities, lpips_corrs):
        ax2.annotate(f'{corr:.2f}', (s, corr), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_plots.png', dpi=300)
    plt.show()

def create_box_plots(df, output_dir):
    """Create distributions of metrics with CAPITALIZED Y-Axis Labels"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Defined Titles as ALL CAPS: DDS, LPIPS, MAP
    metrics = [('dds', 'Blues', 'DDS'), ('lpips', 'Reds', 'LPIPS'), ('mAP', 'Greens', 'MAP')]
    
    for ax, (col, pal, title) in zip(axes, metrics):
        if col in df.columns:
            sns.boxplot(data=df, x='severity', y=col, ax=ax, palette=pal, width=0.6)
            sns.stripplot(data=df, x='severity', y=col, ax=ax, color='black', alpha=0.5, jitter=True)
            
            ax.set_title(f'{title} Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Severity', fontsize=12)
            
            # --- FORCE Y-LABEL TO BE CAPITALIZED ---
            ax.set_ylabel(title, fontsize=12, fontweight='bold')
            
        else:
            ax.text(0.5, 0.5, f'No {title} Data', ha='center')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'box_plots.png', dpi=300)
    plt.show()

def main():
    # --- CONFIGURATION ---
    input_json = "VOC_results.json"
    output_dir = Path("plots/VOC_plots")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading {input_json}...")
    try:
        data = load_corruption_data(input_json)
    except FileNotFoundError:
        print(f"Error: File {input_json} not found!")
        return

    print("Extracting metrics...")
    df = extract_metrics_for_plots(data)
    
    if df.empty:
        print("No valid data found to plot.")
        return

    print(f"Found {len(df)} data points across {df['severity'].nunique()} severity levels.")
    
    # Compute Correlations
    corrs = compute_correlations_by_severity(df)
    
    # Generate Plots
    print("Generating plots...")
    create_correlation_plots(corrs, output_dir)
    create_box_plots(df, output_dir)
    
    print(f"Done! Plots saved to {output_dir}/")

if __name__ == '__main__':
    main()