import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


# Global plotting font sizes (applies to all plots in this file)
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})


def plot_aggregate_results(parquet_path: str):
    """Plots the mean recovery curve across the entire dataset."""
    df = pd.read_parquet(parquet_path)
    
    # Identify layer columns (layer_00 to layer_23)
    layer_cols = [f"layer_{i:02d}" for i in range(35)]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    means = df[layer_cols].mean()
    stds = df[layer_cols].std()

    plt.plot(range(35), means.values, marker='o', color='#3498db', label='Mean Recovery')
    plt.fill_between(range(35), means - stds, means + stds, alpha=0.2, color='#3498db', label='Std Dev')
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Transformer Layer')
    plt.ylabel('Causal Recovery Ratio')
    plt.title(f'Aggregate Causal Tracing (N={len(df)})')
    plt.xticks(range(0, 35, 2))
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.savefig("aggregate_recovery.png", dpi=300)
    print("Saved aggregate_recovery.png")

def plot_single_trace(parquet_path, sample_idx):
    """Plots the recovery curve for a specific sample index."""
    df = pd.read_parquet(parquet_path)
    
    # Locate the specific sample
    # row = df[df['idx'] == sample_idx]
    # if row.empty:
    #     print(f"Index {sample_idx} not found. Available indices: {df['idx'].unique()[:10]}...")
    #     return
    row = df.iloc[0]
    
    layers = [f"layer_{i:02d}" for i in range(35)]
    recovery_values = [row[l] for l in layers]
    
    # Calculate Persistence Layer (L_p)
    persistence_layer = 24
    for i, val in enumerate(recovery_values):
        if val >= 0.5:
            persistence_layer = i
            break

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(range(35), recovery_values, marker='o', color='#2ecc71', linewidth=2.5, label='Recovery')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.6)
    plt.axvline(x=persistence_layer, color='orange', linestyle=':', alpha=0.8)
    
    # Metadata and Annotations
    plt.title(f"Causal Trace: {row['instruction']}\n(Episode {row['episode_idx']}, Frame {row['frame_idx']})", fontsize=12)
    plt.xlabel("Layer", fontsize=10)
    plt.ylabel("Recovery Ratio", fontsize=10)
    plt.xticks(range(0, 35, 2))
    plt.ylim(-0.05, 1.05)
    
    # stats_text = (
    #     f"L_p: {persistence_layer}\n"
    #     # f"Mean Entropy: {row['entropy_mean']:.3f}\n"
    #     f"V1 Entropy: {row['v1_entropy']:.3f}\n"
    #     f"V8 Entropy: {row['v8_entropy']:.3f}"
    # )
    plt.gca().text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, 
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(f"trace_idx_{sample_idx}.png", dpi=300)
    print(f"Saved trace_idx_{sample_idx}.png")

def plot_token_entropies(parquet_path, sample_idx):
    """Plots the entropy across the 56-token horizon for a single sample."""
    df = pd.read_parquet(parquet_path)
    # row = df[df['idx'] == sample_idx]
    # if row.empty: return
    row = df.iloc[0]
    
    clean_entropies = row['clean_token_entropies'] 
    corrupted_entropies = row['corrupted_token_entropies']
    clean_max_probs = row['clean_max_probs']
    corrupted_max_probs = row['corrupted_max_probs']
    
    print(f"Clean Entropies: {clean_entropies}")
    print(f"Corrupted Entropies: {corrupted_entropies}")
    print(f"Clean Max Probs: {clean_max_probs}")
    print(f"Corrupted Max Probs: {corrupted_max_probs}")

    # If the list is stored as a string (happens in some Parquet engines), convert back
    if isinstance(clean_entropies, str):
        import json
        clean_entropies = json.loads(clean_entropies)
    if isinstance(corrupted_entropies, str):
        corrupted_entropies = json.loads(corrupted_entropies)

    plt.figure(figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, 8))
    
    for i in range(8):
        start, end = i * 7, (i + 1) * 7
        plt.bar(range(start, end), entropies[start:end], color=colors[i], alpha=0.7, edgecolor='black')
        if i < 7: plt.axvline(x=end - 0.5, color='gray', linestyle='--', alpha=0.2)

    plt.title(f"Action Entropy Horizon (Sample {sample_idx}):\n{row['instruction']}")
    plt.xlabel("Action Dimension (8 timesteps x 7 dims)")
    plt.ylabel("Entropy (bits)")
    plt.xticks(range(0, 57, 7))
    
    # Label the x, y, z dimensions for each vector
    dim_labels = ['x', 'y', 'z', 'r', 'p', 'y', 'g']
    for i in range(56):
        plt.text(i, -max(entropies)*0.05, dim_labels[i % 7], ha='center', fontsize=7, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"entropy_horizon_{sample_idx}.png", dpi=300)
    print(f"Saved entropy_horizon_{sample_idx}.png")

def analyze_snapshot(parquet_path):
    # 1. Load Data
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    # 2. Extract Data (Mapping lists to scalars)
    # Using 'clean_token_entropies' as specified
    print("Computing Joint Entropy (Sum) across 56-token horizon...")
    # print(df['clean_token_entropies'].head())  # Debug: Check the structure of the column
    df['total_entropy'] = df['clean_token_entropies'].apply(
        lambda x: np.sum(x) if x is not None and len(x) > 0 else np.nan
    )

    # print(df['total_entropy'].head())  # Debug: Check the computed total entropy
    print(df.columns)
    # 3. Compute Persistence Layer (L_p) 
    # The first layer where recovery >= 0.5
    layer_cols = [f"layer_{i:02d}" for i in range(35)]
    print(df[layer_cols].head())  # Debug: Check the layer columns

    def get_lp(row):
        for i, col in enumerate(layer_cols):
            if row[col] <= 0.8:
                return i
        return 24 # Didn't hit threshold
        
    df['lp'] = df.apply(get_lp, axis=1)

    # 4. Filter for successful traces
    # We remove '24' so we don't skew the correlation with failed recoveries
    clean_df = df[df['lp'] < 24].dropna(subset=['total_entropy'])
    # print(f"Analyzing {len(clean_df)} valid traces.")
    # print(clean_df[['total_entropy', 'lp']].head())  # Debug: Check the final DataFrame for correlation
    # 5. Statistical Calculation
    r, p = stats.pearsonr(clean_df['total_entropy'], clean_df['lp'])
    print(f"\n--- 30% Progress Result ---")
    print(f"Pearson Correlation (r): {r:.4f}")
    print(f"P-value:                 {p:.4e}")
    print(f"---------------------------\n")

    # 6. Plotting
    plt.figure(figsize=(14, 6))
    sns.set_style("white")

    # Plot A: KDE Contour (Better for 20,000+ points)
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=clean_df, x='total_entropy', y='lp', cmap="Blues", fill=True, alpha=0.6)
    sns.regplot(data=clean_df, x='total_entropy', y='lp', scatter=False, color="#e74c3c")
    plt.title(f"30% Check: Ambiguity vs Depth\nr = {r:.3f}", fontsize=14)
    plt.xlabel("Joint Entropy (Sum of 56 tokens)", fontsize=12)
    plt.ylabel("Persistence Layer (L_p)", fontsize=12)
    plt.ylim(0, 23)

    # Plot B: Boxplot of Entropy per Layer
    plt.subplot(1, 2, 2)
    sns.boxplot(data=clean_df, x='lp', y='total_entropy', palette='Spectral', showfliers=False)
    plt.title("Is Entropy Higher for Deeper Layers?", fontsize=14)
    plt.xlabel("Persistence Layer (L_p)", fontsize=12)
    plt.ylabel("Total Entropy (bits)", fontsize=12)

    plt.tight_layout()
    plt.savefig("check_30_percent_results.png", dpi=300)
    print("Plot saved to 'check_30_percent_results.png'.")


def plot_lp_vs_time(path, threshold=0.8):
    # 1. Load Data
    
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} samples.")

    # 2. Identify Layers and Calculate L_p (Persistence Layer)
    layer_cols = [f"layer_{i:02d}" for i in range(35)]
    
    # Check if any recovery hits the threshold
    # L_p is defined as the last layer where recovery >= threshold
    layer_values = df[layer_cols].to_numpy(dtype=float, na_value=-np.inf)
    mask = layer_values >= threshold
    has_recovered = mask.any(axis=1)

    # Use reversed argmax to find the last index where threshold is reached
    # We assign len(layer_cols) to those that never reached the threshold
    fail_sentinel = len(layer_cols)
    last_hit_idx = (len(layer_cols) - 1) - np.argmax(mask[:, ::-1], axis=1)
    df['lp'] = np.where(has_recovered, last_hit_idx, fail_sentinel)

    # 3. Aggregate by Frame Index
    # We ignore the failures (lp=fail_sentinel) for the mean to see the internal model trend
    time_df = (
        df[df['lp'] < fail_sentinel]
        .groupby('frame_idx')['lp']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .sort_values('frame_idx')
    )
    time_df['std'] = time_df['std'].fillna(0.0)

    if time_df.empty:
        print(f"No samples reached threshold={threshold} at any frame.")
        return

    # 4. Plotting the Mean Curve with Standard Deviation Band
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    ax1 = plt.gca()
    ax1.plot(
        time_df['frame_idx'],
        time_df['mean'],
        color='#2980b9',
        lw=2.5,
        label='Mean Persistence Layer (L_p)'
    )
    lower = np.clip(time_df['mean'] - time_df['std'], 0, 23)
    upper = np.clip(time_df['mean'] + time_df['std'], 0, 23)
    ax1.fill_between(
        time_df['frame_idx'],
        lower,
        upper,
        color='#2980b9',
        alpha=0.2,
        label='±1 Std Dev'
    )
    ax1.set_xlabel("Frame Index (Timestep in Episode)", fontsize=12)
    ax1.set_ylabel("Mean Persistence Layer Index", fontsize=12, color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_xlim(time_df['frame_idx'].min(), time_df['frame_idx'].max())
    ax1.set_ylim(0, 23)

    plt.title(f"Mean Recovery Layer with Std Dev Band\n(Threshold={threshold} Over Time)", fontsize=14)
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("lp_vs_time_mean_std.png", dpi=300)
    print("Plot saved as 'lp_vs_time_mean_std.png'")

    # 5. Summary Stats
    early_mean = time_df[time_df['frame_idx'] < 10]['mean'].mean()
    late_mean = time_df[time_df['frame_idx'] > 40]['mean'].mean()
    print(f"Early Depth (Frames < 10): {early_mean:.2f}")
    print(f"Late Depth (Frames > 40):  {late_mean:.2f}")

def compare_difficulty_curves(path):
    # 1. Load snapshot
    df = pd.read_parquet(path)
    
    # 2. Calculate Joint Entropy (The "Difficulty" Metric)
    df['total_entropy'] = df['clean_token_entropies'].apply(np.sum)
    
    # 3. Define the "Hard" and "Easy" subsets (Top/Bottom 10%)
    threshold_low = df['total_entropy'].quantile(0.10)
    threshold_high = df['total_entropy'].quantile(0.90)
    
    easy_df = df[df['total_entropy'] <= threshold_low]
    hard_df = df[df['total_entropy'] >= threshold_high]
    
    print(f"Easy samples (Entropy < {threshold_low:.2f}): {len(easy_df)}")
    print(f"Hard samples (Entropy > {threshold_high:.2f}): {len(hard_df)}")

    # 4. Extract Mean Recovery Curves
    layer_cols = [f"layer_{i:02d}" for i in range(35)]
    easy_curve = easy_df[layer_cols].mean()
    hard_curve = hard_df[layer_cols].mean()
    
    # Standard Error for shading
    easy_sem = easy_df[layer_cols].sem()
    hard_sem = hard_df[layer_cols].sem()

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Easy Curve
    plt.plot(range(35), easy_curve, label='Easiest 10% (Low Ambiguity)', color='#2ecc71', lw=3)
    plt.fill_between(range(35), easy_curve - easy_sem, easy_curve + easy_sem, color='#2ecc71', alpha=0.2)

    # Hard Curve
    plt.plot(range(35), hard_curve, label='Hardest 10% (High Ambiguity)', color='#e74c3c', lw=3)
    plt.fill_between(range(35), hard_curve - hard_sem, hard_curve + hard_sem, color='#e74c3c', alpha=0.2)

    # Styling
    plt.title("Language Recovery Profile: Hard vs. Easy Tasks", fontsize=14)
    plt.xlabel("Model Layer", fontsize=12)
    plt.ylabel("Mean Recovery Ratio", fontsize=12)
    # plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Recovery Threshold (0.5)')
    plt.xticks(range(0, 35, 2))
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.savefig("difficulty_curve_comparison.png", dpi=300)
    print("Plot saved to 'difficulty_curve_comparison.png'")

    # 6. Quantify the "Depth Shift"
    threshold = 0.8
    easy_lp = np.argmax(easy_curve.values < threshold) if any(easy_curve < threshold) else 24
    hard_lp = np.argmax(hard_curve.values < threshold) if any(hard_curve < threshold) else 24
    print(f"\nAggregate L_p (Easy): Layer {easy_lp}")
    print(f"Aggregate L_p (Hard): Layer {hard_lp}")
    print(f"Net Depth Shift: {hard_lp - easy_lp} layers")

def plot_recovery_comparison(language_path, action_path):
    """Plot mean language recovery and action recovery rates over layers with std dev."""
    df_lang = pd.read_parquet(language_path)
    df_action = pd.read_parquet(action_path)
    
    layer_cols = [f"layer_{i:02d}" for i in range(35)]
    
    # Calculate mean and standard deviation
    lang_mean = df_lang[layer_cols].mean()
    lang_std = df_lang[layer_cols].std()
    
    action_mean = df_action[layer_cols].mean()
    action_std = df_action[layer_cols].std()
    
    # Create plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Plot language recovery
    plt.plot(range(35), lang_mean, marker='o', color='#3498db', lw=2.5, 
             label='Language Recovery')
    plt.fill_between(range(35), lang_mean - lang_std, lang_mean + lang_std, 
                     color='#3498db', alpha=0.2)
    
    # Plot action recovery
    plt.plot(range(35), action_mean, marker='s', color='#e74c3c', lw=2.5, 
             label='Action Recovery')
    plt.fill_between(range(35), action_mean - action_std, action_mean + action_std, 
                     color='#e74c3c', alpha=0.2)
    
    # Benchmarks
    # plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Recovery Threshold')
    # plt.axhline(y=0.0, color='black', lw=1)
    
    plt.title("Causal Trace: Language vs Action Recovery Across Layers", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Mean Recovery Ratio", fontsize=12)
    plt.xticks(range(0, 35, 2))
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig("recovery_comparison.png", dpi=300)
    print("Plot saved as 'recovery_comparison.png'")
    
    # Print summary statistics
    print(f"\n--- Recovery Comparison ---")
    print(f"Language - Peak: {lang_mean.max():.4f} at layer {lang_mean.idxmax()}")
    print(f"Action - Peak: {action_mean.max():.4f} at layer {action_mean.idxmax()}")

def analyze_momentum_signal(path):
    df = pd.read_parquet(path)
    layer_cols = [f"layer_{i:02d}" for i in range(35)]
    
    # 1. Calculate the Mean Recovery across all layers
    # Even if individual samples are 0, the mean might show a 'hump'
    mean_recovery = df[layer_cols].mean()
    std_error = df[layer_cols].sem()

    # 2. Plotting the Causal Importance of V1 (Momentum)
    plt.figure(figsize=(10, 6))
    sns.set_style("white")
    
    # The 'Signal' Line
    plt.plot(range(35), mean_recovery, marker='o', color='#8e44ad', lw=2.5, label='V1 Momentum Recovery')
    plt.fill_between(range(35), mean_recovery - std_error, mean_recovery + std_error, color='#8e44ad', alpha=0.2)
    
    # Benchmarks
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Standard Recovery Threshold')
    plt.axhline(y=0.0, color='black', lw=1)

    plt.title("Causal Trace: Importance of Previous Actions ($V_1$) on Future Plan", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Mean Recovery Ratio", fontsize=12)
    plt.xticks(range(0, 35, 2))
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.savefig("v1_momentum_trace.png", dpi=300)
    print("Plot saved as 'v1_momentum_trace.png'")

    # 3. Print the peak signal
    peak_layer = mean_recovery.idxmax()
    peak_val = mean_recovery.max()
    print(f"\n--- Momentum Results ---")
    print(f"Peak Signal Strength: {peak_val:.4f} at {peak_layer}")
    print(f"Comparison: Language Recovery was ~0.8-0.9. Momentum is {peak_val/0.8:.1%} as strong.")


def plot_fusion_curves(attention_path):
    df = pd.read_csv(attention_path)
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Color mapping: green shades, blue shades, red shades (dark to light)
    colors = ["#164E11", "#110E6C", "#1A1167",  # green shades
              "#992153", "#61A5ED", "#b80dd2",   # blue shades
              "#ACB111", "#f00707", "#EB9411"]   # red shades
    

    linestyles = ['-', '--', ':',
                  '--', '-', '-.',
                ':', '-.', '-',
                ]  # Different line styles for each group

    for idx, col in enumerate(df.columns[1:]):
        color = colors[idx] if idx < len(colors) else '#cccccc'
        plt.plot(df['layer'], df[col], marker='o', linewidth=2, markersize=6, label=" ".join(col.split("_")), color=color, linestyle=linestyles[idx])

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Attention Weight", fontsize=12)
    plt.title("Attention Curves", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig("attention_curves.png", dpi=300)
    print("Plot saved as 'attention_curves.png'")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, required=True, help="Path to results.parquet")
    # parser.add_argument("--idx", type=int, required=True, help="Index of sample to plot")
    # parser.add_argument("--aggregate", action="store_true", help="Plot mean of all results")
    # args = parser.parse_args()
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))

    action_path = os.path.join(project_root, "data/data/tracing_outputs/action_tracing_results_final.parquet")
    language_path = os.path.join(project_root, "data/data/tracing_outputs/language_tracing_results_final.parquet")
    plot_recovery_comparison(language_path, action_path)
    compare_difficulty_curves(language_path)
    plot_lp_vs_time(language_path)
    # plot_token_entropies(args.path, args.idx)
    # plot_single_trace(args.path, args.idx)
    # attention_path = os.path.join(project_root, "data/data/activation_outputs/attention_density.csv")
    # plot_fusion_curves(attention_path)