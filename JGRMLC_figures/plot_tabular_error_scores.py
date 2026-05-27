import matplotlib.pyplot as plt
import numpy as np

# Data from Table 2
test_cases = ['Validation (5400s)', 'Condensation (4800s)',
              'Sedimentation (7200s)', 'RICO']
metrics = ['KL Divergence', 'Wasserstein', 'Total Mass MAE']

# Model data - colors: orange, red, green
colors = ['#ff7f0e', '#d62728', '#2ca02c']
model_names = ['AE-SINDy', 'AE-NNddzdt', 'AE-AR']

# Hatching patterns for different test cases
hatches = ['', '///', '\\\\\\', 'xxx']

# Data organized by test case, then metric, then model
data = {
    'Validation (5400s)': {
        'KL Divergence': [7.27e-3, 5.27e-3, 4.76e-3],
        'Wasserstein': [4.27e-3, 4.41e-3, 4.39e-3],
        'Total Mass MAE': [6.25e-9, 2.88e-2, 2.18e-2]
    },
    'Condensation (4800s)': {
        'KL Divergence': [1.24e-2, 1.13e-2, 1.25e-2],
        'Wasserstein': [6.18e-3, 6.91e-3, 7.62e-3],
        'Total Mass MAE': [4.51e-9, 3.43e-2, 3.70e-2]
    },
    'Sedimentation (7200s)': {
        'KL Divergence': [8.91e-3, 9.78e-3, 9.78e-3],
        'Wasserstein': [6.67e-3, 6.41e-3, 7.50e-3],
        'Total Mass MAE': [1.41e-8, 3.80e-2, 2.84e-2]
    },
    'RICO': {
        'KL Divergence': [6.58e-3, 7.65e-3, 1.02e-2],
        'Wasserstein': [6.61e-3, 6.90e-3, 8.22e-3],
        'Total Mass MAE': [1.03e-9, 2.08e-2, 2.82e-2]
    }
}

# Create figure with 1:1 aspect ratio
fig, ax = plt.subplots(figsize=(8, 5))

# Number of groups (metrics) and bars per group
n_metrics = len(metrics)
n_models = len(model_names)
n_tests = len(test_cases)

# Bars per metric = n_models (grouped together) with n_tests bars each
bars_per_model = n_tests
group_width = n_models * (bars_per_model + 0.5)  # Space between models within a metric

# x positions for metric groups
metric_positions = np.arange(n_metrics) * (group_width + 2)

# Width of individual bars
bar_width = 0.7

# Plot bars
for metric_idx, metric in enumerate(metrics):
    base_x = metric_positions[metric_idx]

    for model_idx, model_name in enumerate(model_names):
        for test_idx, test_case in enumerate(test_cases):
            # Calculate x position: group by model, then test cases side-by-side
            x_pos = base_x + model_idx * (bars_per_model + 0.5) + test_idx

            # Get value
            value = data[test_case][metric][model_idx]

            # Plot bar
            bar = ax.bar(x_pos, value, bar_width,
                         color=colors[model_idx],
                         hatch=hatches[test_idx],
                         edgecolor='black',
                         linewidth=0.5,
                         alpha=0.8)

            # Bold outline for best value across all models for this test case
            all_values_for_test = data[test_case][metric]
            if value == min(all_values_for_test):
                bar[0].set_linewidth(2.5)

# Set x-axis labels at metric centers
metric_centers = [pos + group_width / 2 - 0.5 for pos in metric_positions]
ax.set_xticks(metric_centers)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')

# Add vertical lines to separate metrics
for pos in metric_positions[1:]:
    ax.axvline(x=pos - 1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# Add subtle lines to separate models within metrics
for metric_idx in range(n_metrics):
    base_x = metric_positions[metric_idx]
    for model_idx in range(1, n_models):
        sep_x = base_x + model_idx * (bars_per_model + 0.5) - 0.25
        ax.axvline(x=sep_x, color='lightgray', linestyle=':', linewidth=1, alpha=0.5)

# Labels and formatting
ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison Across Test Cases and Metrics',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
#ax.set_yscale('log')

# Create custom legends
from matplotlib.patches import Patch

# Model legend (colors)
model_legend_elements = [Patch(facecolor=colors[i], edgecolor='black',
                               label=model_names[i], alpha=0.8)
                         for i in range(n_models)]

# Test case legend (hatches)
test_legend_elements = [Patch(facecolor='gray', edgecolor='black',
                              hatch=hatches[i], label=test_cases[i], alpha=0.6)
                        for i in range(n_tests)]

# Add legends
legend1 = ax.legend(handles=model_legend_elements, title='Model',
                    loc='upper left', fontsize=10, title_fontsize=11)
legend2 = ax.legend(handles=test_legend_elements, title='Test Case',
                    loc='center left', fontsize=10, title_fontsize=11)
ax.add_artist(legend1)  # Add back the first legend

plt.tight_layout()
plt.show()
plt.savefig('tabular_error_scores.pdf')