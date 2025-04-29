"""
This script analyzes the CSV files containing training metrics for ResNet and WRN models
with different approximation methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mtick

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

# Preprocessing to handle null values
for name, df in data_dict.items():
    # Replace NaN with None for better visualization
    data_dict[name] = df.replace([np.inf, -np.inf], np.nan)
    
    # Print information about null values
    null_count = df['log_marglik'].isna().sum()
    total_count = len(df)
    print(f"{name}: {null_count}/{total_count} null values in log_marglik column")

# Part 1: Final performance analysis - comparing approximation methods
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
            'train_nll': final_row['train_nll'],
            'valid_nll': final_row['valid_nll'],
            'log_marglik': final_row['log_marglik'],
            'gen_gap': final_row['train_perf'] - final_row['valid_perf']
        })
    
    # Convert to DataFrame
    final_df = pd.DataFrame(final_metrics)
    
    # Print summary table
    print("\nFinal Performance Summary:")
    print(final_df[['name', 'train_perf', 'valid_perf', 'gen_gap', 'log_marglik']])
    
    # Plot final validation performance
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='approx', y='valid_perf', hue='model', data=final_df)
    plt.title('Final Validation Performance by Model and Approximation Method')
    plt.ylabel('Validation Performance')
    plt.xlabel('Approximation Method')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(title='Model Architecture')
    plt.savefig('final_valid_perf.png', dpi=300, bbox_inches='tight')
    
    # Plot generalization gap
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='approx', y='gen_gap', hue='model', data=final_df)
    plt.title('Generalization Gap by Model and Approximation Method')
    plt.ylabel('Generalization Gap (Train - Valid)')
    plt.xlabel('Approximation Method')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(title='Model Architecture')
    plt.savefig('gen_gap.png', dpi=300, bbox_inches='tight')
    
    # Plot log marginal likelihood
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='log_marglik', hue='model', data=final_df)
    plt.title('Log Marginal Likelihood by Model and Approximation Method')
    plt.ylabel('Log Marginal Likelihood')
    plt.xlabel('Approximation Method')
    plt.legend(title='Model Architecture')
    plt.savefig('log_marglik.png', dpi=300, bbox_inches='tight')
    
    return final_df

# Part 2: Convergence analysis - evaluating training dynamics
def analyze_convergence():
    # Plot training curves for validation performance
    plt.figure(figsize=(14, 8))
    
    # Define line styles and colors for better visualization
    linestyles = {
        'resnet': '-',
        'wrn': '--'
    }
    
    colors = {
        'diag': 'blue',
        'kernel': 'green',
        'kron': 'red'
    }
    
    for name, df in data_dict.items():
        model, approx = name.split('_')
        plt.plot(
            df['epoch'], 
            df['valid_perf'], 
            label=f"{model}_{approx}",
            linestyle=linestyles[model],
            color=colors[approx],
            linewidth=2.5 if model == 'wrn' else 2
        )
    
    plt.title('Validation Performance Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('valid_perf_convergence.png', dpi=300, bbox_inches='tight')
    
    # Plot training performance
    plt.figure(figsize=(14, 8))
    
    for name, df in data_dict.items():
        model, approx = name.split('_')
        plt.plot(
            df['epoch'], 
            df['train_perf'], 
            label=f"{model}_{approx}",
            linestyle=linestyles[model],
            color=colors[approx],
            linewidth=2.5 if model == 'wrn' else 2
        )
    
    plt.title('Training Performance Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_perf_convergence.png', dpi=300, bbox_inches='tight')
    
    # Plot log marginal likelihood with proper handling of NaN values
    plt.figure(figsize=(14, 8))
    
    for name, df in data_dict.items():
        model, approx = name.split('_')
        # Drop NaN values
        valid_data = df.dropna(subset=['log_marglik'])
        if not valid_data.empty:
            plt.plot(
                valid_data['epoch'], 
                valid_data['log_marglik'], 
                label=f"{model}_{approx}",
                linestyle=linestyles[model],
                color=colors[approx],
                linewidth=2.5 if model == 'wrn' else 2
            )
    
    plt.title('Log Marginal Likelihood Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Log Marginal Likelihood')
    plt.legend()
    plt.grid(True)
    plt.savefig('log_marglik_convergence.png', dpi=300, bbox_inches='tight')
    
    # Individual plots for log_marglik (one per model-approx combination)
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
    
    # Compare convergence speed between models
    convergence_epochs = {}
    
    for name, df in data_dict.items():
        final_valid = df['valid_perf'].iloc[-1]
        threshold = 0.9 * final_valid
        
        convergence_epoch = None
        for i, perf in enumerate(df['valid_perf']):
            if perf >= threshold:
                convergence_epoch = i
                break
        
        convergence_epochs[name] = convergence_epoch
    
    print("\nConvergence Analysis (epochs to reach 90% of final performance):")
    for name, epoch in convergence_epochs.items():
        print(f"{name}: {epoch}")
    
    # Create convergence speed dataframe
    conv_data = []
    for name, epoch in convergence_epochs.items():
        if epoch is not None:  # Handle case where threshold might not be reached
            model, approx = name.split('_')
            conv_data.append({
                'name': name,
                'model': model,
                'approx': approx,
                'convergence_epoch': epoch
            })
    
    if conv_data:  # Only create plot if there's data
        conv_df = pd.DataFrame(conv_data)
        
        # Plot convergence speed
        plt.figure(figsize=(12, 6))
        sns.barplot(x='approx', y='convergence_epoch', hue='model', data=conv_df)
        plt.title('Epochs to Convergence by Model and Approximation Method')
        plt.ylabel('Epochs to Reach 90% of Final Performance')
        plt.xlabel('Approximation Method')
        plt.legend(title='Model Architecture')
        plt.savefig('convergence_speed.png', dpi=300, bbox_inches='tight')
        
        return conv_df
    
    return None

# Part 3: Width effect analysis - comparing ResNet vs WRN
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
        
        # For log_marglik, check if values exist
        marglik_diff = None
        if not pd.isna(resnet_final['log_marglik']) and not pd.isna(wrn_final['log_marglik']):
            marglik_diff = wrn_final['log_marglik'] - resnet_final['log_marglik']
        
        width_effect.append({
            'approx': approx,
            'resnet_perf': resnet_final['valid_perf'],
            'wrn_perf': wrn_final['valid_perf'],
            'abs_improvement': perf_improvement,
            'rel_improvement': relative_improvement,
            'resnet_marglik': resnet_final['log_marglik'],
            'wrn_marglik': wrn_final['log_marglik'],
            'marglik_diff': marglik_diff
        })
    
    width_df = pd.DataFrame(width_effect)
    
    print("\nWidth Effect Analysis:")
    print(width_df[['approx', 'resnet_perf', 'wrn_perf', 'abs_improvement', 'rel_improvement']])
    
    # Plot absolute performance improvement
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='approx', y='abs_improvement', data=width_df)
    plt.title('Absolute Performance Improvement from Increased Width')
    plt.ylabel('WRN - ResNet Validation Performance')
    plt.xlabel('Approximation Method')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.savefig('width_abs_improvement.png', dpi=300, bbox_inches='tight')
    
    # Plot relative performance improvement
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='approx', y='rel_improvement', data=width_df)
    plt.title('Relative Performance Improvement (%) from Increased Width')
    plt.ylabel('Percentage Improvement')
    plt.xlabel('Approximation Method')
    plt.savefig('width_rel_improvement.png', dpi=300, bbox_inches='tight')
    
    # Plot log marginal likelihood difference if available
    if width_df['marglik_diff'].notna().any():
        plt.figure(figsize=(12, 6))
        sns.barplot(x='approx', y='marglik_diff', data=width_df.dropna(subset=['marglik_diff']))
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

# Part 4: Neural Tangent Kernel (NTK) regime analysis
def analyze_ntk_regime():
    """
    Analyze how well the models align with the Neural Tangent Kernel (NTK) regime predictions.
    In the NTK regime, wider networks should:
    1. Converge faster
    2. Have better generalization
    3. Show more deterministic behavior
    """
    # Compare convergence rates between ResNet and WRN
    convergence_rates = []
    
    for approx in ['diag', 'kernel', 'kron']:
        resnet_data = data_dict[f'resnet_{approx}']
        wrn_data = data_dict[f'wrn_{approx}']
        
        # Calculate average improvement in validation performance per epoch
        resnet_valid_diff = [resnet_data['valid_perf'][i] - resnet_data['valid_perf'][i-1] 
                            for i in range(1, len(resnet_data))]
        wrn_valid_diff = [wrn_data['valid_perf'][i] - wrn_data['valid_perf'][i-1] 
                          for i in range(1, len(wrn_data))]
        
        # Calculate average improvement in early epochs (e.g., first 20 epochs)
        early_epochs = min(20, len(resnet_valid_diff))
        resnet_early_rate = sum(resnet_valid_diff[:early_epochs]) / early_epochs
        wrn_early_rate = sum(wrn_valid_diff[:early_epochs]) / early_epochs
        
        convergence_rates.append({
            'approx': approx,
            'resnet_early_rate': resnet_early_rate,
            'wrn_early_rate': wrn_early_rate,
            'rate_improvement': wrn_early_rate - resnet_early_rate,
            'relative_improvement': (wrn_early_rate - resnet_early_rate) / abs(resnet_early_rate) * 100 if resnet_early_rate != 0 else 0
        })
    
    conv_rate_df = pd.DataFrame(convergence_rates)
    
    print("\nNTK Regime Analysis - Convergence Rates:")
    print(conv_rate_df)
    
    # Plot convergence rate improvement
    plt.figure(figsize=(12, 6))
    sns.barplot(x='approx', y='rate_improvement', data=conv_rate_df)
    plt.title('Convergence Rate Improvement from Increased Width')
    plt.ylabel('WRN - ResNet Average Improvement Per Epoch')
    plt.xlabel('Approximation Method')
    plt.savefig('ntk_convergence_rate.png', dpi=300, bbox_inches='tight')
    
    # Analyze generalization gap between ResNet and WRN
    gen_gap = []
    
    for approx in ['diag', 'kernel', 'kron']:
        resnet_data = data_dict[f'resnet_{approx}']
        wrn_data = data_dict[f'wrn_{approx}']
        
        # Get final metrics
        resnet_final = resnet_data.iloc[-1]
        wrn_final = wrn_data.iloc[-1]
        
        # Calculate generalization gap
        resnet_gap = resnet_final['train_perf'] - resnet_final['valid_perf']
        wrn_gap = wrn_final['train_perf'] - wrn_final['valid_perf']
        
        gen_gap.append({
            'approx': approx,
            'resnet_gap': resnet_gap,
            'wrn_gap': wrn_gap,
            'gap_improvement': resnet_gap - wrn_gap,  # Lower gap is better
            'relative_improvement': (resnet_gap - wrn_gap) / resnet_gap * 100 if resnet_gap != 0 else 0
        })
    
    gen_gap_df = pd.DataFrame(gen_gap)
    
    print("\nNTK Regime Analysis - Generalization Gap:")
    print(gen_gap_df)
    
    # Plot generalization gap improvement
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='approx', y='gap_improvement', data=gen_gap_df)
    plt.title('Generalization Gap Improvement from Increased Width')
    plt.ylabel('ResNet Gap - WRN Gap (Positive is Better)')
    plt.xlabel('Approximation Method')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.savefig('ntk_generalization_gap.png', dpi=300, bbox_inches='tight')
    
    return conv_rate_df, gen_gap_df

# Define main function to run all analyses
def main():
    print("Starting analysis...")
    
    final_df = analyze_final_performance()
    conv_df = analyze_convergence()
    width_df = analyze_width_effect()
    ntk_conv_df, ntk_gen_df = analyze_ntk_regime()
    
    print("Analysis complete. Check the current directory for graph images.")

if __name__ == "__main__":
    main()

