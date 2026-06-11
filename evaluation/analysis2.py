import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configuration ---
FILE_PATH = "Cross Robot Evalution (Responses) - Form Responses 1.csv"
OUTPUT_DIR = "publication_plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 3 phrase prefixes — responses are combined across all 3 for each question
PREFIXES = ["I1", "I2", "I3"]

# The exact 7-point Likert scale used in the Google Form
SCALE_ORDER = [
    "Strongly Disagree",
    "Disagree",
    "Somewhat Disagree",
    "Neutral",
    "Somewhat Agree",
    "Agree",
    "Strongly Agree"
]
SCALE_MAPPING = {val: i + 1 for i, val in enumerate(SCALE_ORDER)}

# --- 2. Data Loading & Cleaning ---
df = pd.read_csv(FILE_PATH)

# Strip any trailing whitespace from column headers
df.columns = df.columns.str.strip()

# Dynamically extract the 4 distinct Likert questions by referencing the 'I1' columns
base_questions = [col.split(': ', 1)[1] for col in df.columns if col.startswith('I1:')]

# --- 3. Plot Generation ---
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

for q_idx, question in enumerate(base_questions):

    # Gather all responses for this question across the 3 phrase videos (I1, I2, I3)
    cols = [f"{p}: {question}" for p in PREFIXES]

    # Flatten into a single list and drop any missing responses
    responses = df[cols].values.flatten()
    responses = [r for r in responses if pd.notna(r)]

    # Count frequencies for each Likert option
    counts = {s: 0 for s in SCALE_ORDER}
    for r in responses:
        if r in counts:
            counts[r] += 1

    # Calculate summary statistics
    numeric_vals = [SCALE_MAPPING[r] for r in responses if r in SCALE_MAPPING]
    mean_val = np.mean(numeric_vals) if numeric_vals else 0
    median_val = np.median(numeric_vals) if numeric_vals else 0

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create the categorical bar chart
    sns.barplot(
        x=SCALE_ORDER,
        y=[counts[s] for s in SCALE_ORDER],
        ax=ax,
        color='steelblue',
        edgecolor='black',
        alpha=0.9
    )

    # Overlay Mean and Median lines (subtract 1 because x-axis indices are 0-6)
    ax.axvline(mean_val - 1, color='firebrick', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val - 1, color='gold', linestyle='-', linewidth=2.5, label=f'Median: {median_val:.0f}')

    # Formatting
    ax.set_title(f"Cross-Robot Evaluation\n{question}", pad=15, weight='bold')
    ax.set_ylabel("Response Count", weight='bold')
    ax.set_xlabel("")
    ax.set_xticks(range(len(SCALE_ORDER)))
    ax.set_xticklabels(SCALE_ORDER, rotation=40, ha='right')

    # Annotate total response count
    ax.text(
        0.98, 0.97, f"n = {len(numeric_vals)}",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color='gray'
    )

    sns.despine()
    ax.legend(frameon=True, loc='upper left')

    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/cross_robot_q{q_idx + 1}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Successfully generated {len(base_questions)} publication-ready plots in the '{OUTPUT_DIR}' directory.")
