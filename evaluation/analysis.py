import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configuration ---
FILE_PATH = "Automatic Gestures From Text (Responses) - Form Responses 1.csv"
OUTPUT_DIR = "publication_plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map the video prefixes to the actual model conditions tested in the study
CONDITIONS = {
    "Hand Animated (Baseline)": ["A1", "A2", "A3"],
    "Llama 3 8B": ["B1", "B2", "B3"],
    "GPT 5.4 Mini": ["C1", "C2", "C3"],
    "Gemini 3.1 Flash Lite": ["D1", "D2", "D3"]
}

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

# Strip any trailing whitespace from column headers (e.g., "C3: ... ")
df.columns = df.columns.str.strip()

# Dynamically extract the 4 distinct Likert questions by referencing the 'A1' columns
base_questions = [col.split(': ', 1)[1] for col in df.columns if col.startswith('A1:')]

# --- 3. Plot Generation ---
# Set a clean, academic visual style
sns.set_theme(style="ticks", context="paper", font_scale=1.2)

for cond_name, prefixes in CONDITIONS.items():
    for q_idx, question in enumerate(base_questions):
        
        # Gather all responses for this specific question across the 3 videos for the current condition
        cols = [f"{p}: {question}" for p in prefixes]
        
        # Flatten into a single array and drop any missing responses (NaNs)
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
        bars = sns.barplot(
            x=SCALE_ORDER, 
            y=[counts[s] for s in SCALE_ORDER], 
            ax=ax, 
            color='steelblue', 
            edgecolor='black',
            alpha=0.9
        )
        
        # Overlay Mean and Median lines (Subtract 1 because x-axis indices are 0-6)
        ax.axvline(mean_val - 1, color='firebrick', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val - 1, color='gold', linestyle='-', linewidth=2.5, label=f'Median: {median_val:.0f}')
        
        # Formatting
        ax.set_title(f"{cond_name}\n{question}", pad=15, weight='bold')
        ax.set_ylabel("Response Count", weight='bold')
        ax.set_xticklabels(SCALE_ORDER, rotation=40, ha='right')
        
        # Clean up borders
        sns.despine()
        ax.legend(frameon=True, loc='upper right')
        
        # Save figure
        plt.tight_layout()
        safe_cond_name = cond_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # e.g., "llama_3_8b_q1.png"
        filename = f"{OUTPUT_DIR}/{safe_cond_name}_q{q_idx+1}.png" 
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"Successfully generated {len(CONDITIONS) * len(base_questions)} publication-ready plots in the '{OUTPUT_DIR}' directory.")