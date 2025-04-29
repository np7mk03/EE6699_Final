"""
Analysis Code for Neural Network Properties: Width and Approximation Methods
This script analyzes the CSV files containing training metrics for ResNet and WRN models
with different approximation methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Function to load the CSV files
def load_data(file_path):
    return pd.read_csv(file_path)

# Load all datasets
files = [
    'resnet_diag.csv',
    'resnet_kernel.csv',
    'resnet_kron.csv',
    'wrn_diag.csv',
    'wrn_kernel.csv',
    'wrn_kron.csv'
]

data_dict = {}
for file in files:
    name = file.replace('.csv', '')
    try:
        data_dict[name] = load_data(file)
        print(f"Loaded {file} successfully")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Extract model type and approximation method
for key in data_dict:
    parts = key.split('_')
    data_dict[key]['model'] = parts[0]
    data_dict[key]['approx'] = parts[1]
    
# After loading the data files, add this preprocessing
for name, df in data_dict.items():
    # Replace NaN with None for better visualization
    data_dict[name] = df.replace([np.inf, -np.inf], np.nan)
    
    # Print information about null values
    null_count = df['log_marglik'].isna().sum()
    total_count = len(df)
    print(f"{name}: {null_count}/{total_count} null values in log_marglik column")

# Part 1: Final performance analysis
def analyze_final_performance():
    final_metrics = []
    
    for name, df in data_dict.items():
        # Get the last epoch
        final_row = df.iloc[-1]
        
        final_metrics.append({
            'name': name,
            'model': final_row['model'],
            'approx': final_row['approx'],
            'train_perf': final_row['train_perf'],
            'valid_perf': final_row['valid_perf'],
            'train_loss': final_row['train_loss'],
            'log_marglik': final_row['log_marglik'],
            'gen_gap': final_row['train_perf'] - final_row['valid_perf']
        })
    
    # Convert to DataFrame
    final_df = pd.DataFrame(final_metrics)
    
    # Print summary table
    print("Final Performance Summary:")
    print(final_df[['name', 'train_perf', 'valid_perf', 'gen_gap', 'log_marglik']])
    
    # Plot final validation performance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='valid_perf', hue='model', data=final_df)
    plt.title('Final Validation Performance by Model and Approximation Method')
    plt.ylabel('Validation Performance')
    plt.xlabel('Approximation Method')
    plt.savefig('final_valid_perf.png', dpi=300, bbox_inches='tight')
    
    # Plot generalization gap
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='gen_gap', hue='model', data=final_df)
    plt.title('Generalization Gap by Model and Approximation Method')
    plt.ylabel('Generalization Gap (Train - Valid)')
    plt.xlabel('Approximation Method')
    plt.savefig('gen_gap.png', dpi=300, bbox_inches='tight')
    
    # Plot log marginal likelihood
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='log_marglik', hue='model', data=final_df)
    plt.title('Log Marginal Likelihood by Model and Approximation Method')
    plt.ylabel('Log Marginal Likelihood')
    plt.xlabel('Approximation Method')
    plt.savefig('log_marglik.png', dpi=300, bbox_inches='tight')
    
    return final_df

def analyze_convergence():
    # Plot training curves for valid_perf and train_perf
    metrics = ['valid_perf', 'train_perf']
    
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        for name, df in data_dict.items():
            plt.plot(df['epoch'], df[metric], label=name)
        
        plt.title(f'{metric} Over Training Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{metric}_convergence.png', dpi=300, bbox_inches='tight')
    
    # Handle log_marglik with null value handling
    plt.figure(figsize=(14, 8))
    
    for name, df in data_dict.items():
        # Filter out null/NaN values
        valid_data = df.dropna(subset=['log_marglik'])
        
        if not valid_data.empty:
            plt.plot(valid_data['epoch'], valid_data['log_marglik'], label=name)
    
    plt.title('Log Marginal Likelihood Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('log_marglik')
    plt.legend()
    plt.grid(True)
    plt.savefig('log_marglik_convergence.png', dpi=300, bbox_inches='tight')
    
    # Individual plots for log_marglik with null handling
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (name, df) in enumerate(data_dict.items()):
        # Filter out null/NaN values
        valid_data = df.dropna(subset=['log_marglik'])
        
        if not valid_data.empty:
            axes[i].plot(valid_data['epoch'], valid_data['log_marglik'])
        
        axes[i].set_title(f'{name} Log Marginal Likelihood')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('log_marglik')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('log_marglik_individual.png', dpi=300, bbox_inches='tight')
    
    # Continue with the rest of the convergence analysis code...
    convergence_epochs = {}
    
    for name, df in data_dict.items():
        final_valid = df['valid_perf'].iloc[-1]
        threshold = 0.9 * final_valid
        
        for i, perf in enumerate(df['valid_perf']):
            if perf >= threshold:
                convergence_epochs[name] = i
                break
    
    print("\nConvergence Analysis (epochs to reach 90% of final performance):")
    for name, epoch in convergence_epochs.items():
        print(f"{name}: {epoch}")
    
    # Create convergence speed dataframe
    conv_data = []
    for name, epoch in convergence_epochs.items():
        model, approx = name.split('_')
        conv_data.append({
            'name': name,
            'model': model,
            'approx': approx,
            'convergence_epoch': epoch
        })
    
    conv_df = pd.DataFrame(conv_data)
    
    # Plot convergence speed
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='convergence_epoch', hue='model', data=conv_df)
    plt.title('Epochs to Convergence by Model and Approximation Method')
    plt.ylabel('Epochs to Reach 90% of Final Performance')
    plt.xlabel('Approximation Method')
    plt.savefig('convergence_speed.png', dpi=300, bbox_inches='tight')
    
    return conv_df
# Part 3: Width effect analysis
def analyze_width_effect():
    # Create a DataFrame to compare ResNet vs WRN for each approximation method
    width_effect = []
    
    for approx in ['diag', 'kernel', 'kron']:
        resnet_data = data_dict[f'resnet_{approx}']
        wrn_data = data_dict[f'wrn_{approx}']
        
        # Get final metrics
        resnet_final = resnet_data.iloc[-1]
        wrn_final = wrn_data.iloc[-1]
        
        # Calculate improvement from increased width
        perf_improvement = wrn_final['valid_perf'] - resnet_final['valid_perf']
        relative_improvement = perf_improvement / resnet_final['valid_perf'] * 100
        
        width_effect.append({
            'approx': approx,
            'resnet_perf': resnet_final['valid_perf'],
            'wrn_perf': wrn_final['valid_perf'],
            'abs_improvement': perf_improvement,
            'rel_improvement': relative_improvement,
            'resnet_marglik': resnet_final['log_marglik'],
            'wrn_marglik': wrn_final['log_marglik'],
            'marglik_diff': wrn_final['log_marglik'] - resnet_final['log_marglik']
        })
    
    width_df = pd.DataFrame(width_effect)
    
    print("\nWidth Effect Analysis:")
    print(width_df[['approx', 'resnet_perf', 'wrn_perf', 'abs_improvement', 'rel_improvement']])
    
    # Plot absolute performance improvement
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='abs_improvement', data=width_df)
    plt.title('Absolute Performance Improvement from Increased Width')
    plt.ylabel('WRN - ResNet Validation Performance')
    plt.xlabel('Approximation Method')
    plt.savefig('width_abs_improvement.png', dpi=300, bbox_inches='tight')
    
    # Plot relative performance improvement
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='rel_improvement', data=width_df)
    plt.title('Relative Performance Improvement (%) from Increased Width')
    plt.ylabel('Percentage Improvement')
    plt.xlabel('Approximation Method')
    plt.savefig('width_rel_improvement.png', dpi=300, bbox_inches='tight')
    
    # Plot log marginal likelihood difference
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='marglik_diff', data=width_df)
    plt.title('Difference in Log Marginal Likelihood from Increased Width')
    plt.ylabel('WRN - ResNet Log Marginal Likelihood')
    plt.xlabel('Approximation Method')
    plt.savefig('width_marglik_diff.png', dpi=300, bbox_inches='tight')
    
    # Plot convergence comparison for each approximation method
    for approx in ['diag', 'kernel', 'kron']:
        plt.figure(figsize=(14, 8))
        
        resnet_data = data_dict[f'resnet_{approx}']
        wrn_data = data_dict[f'wrn_{approx}']
        
        plt.plot(resnet_data['epoch'], resnet_data['valid_perf'], label=f'ResNet {approx}')
        plt.plot(wrn_data['epoch'], wrn_data['valid_perf'], label=f'WRN {approx}')
        
        plt.title(f'Width Effect on Convergence ({approx} approximation)')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'width_convergence_{approx}.png', dpi=300, bbox_inches='tight')
    
    return width_df
if __name__ == "__main__":
    print("Starting analysis...")
    final_df = analyze_final_performance()
    conv_df = analyze_convergence()
    print("Analysis complete. Check the current directory for graph images.")
