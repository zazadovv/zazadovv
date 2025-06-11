import os
import glob
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
import re

# --- GUI to choose directory ---
app = QApplication(sys.argv)
window = QMainWindow()
folderpath = QFileDialog.getExistingDirectory(window, 'Select Folder with all CSVs')
app.quit()

# --- Helpers for sorting ---
def extract_second_e_values(filename):
    e_matches = re.findall(r'E(\d+)', filename)
    if len(e_matches) < 1:
        return float('inf')
    return int(e_matches[1]) if len(e_matches) >= 2 else int(e_matches[0])

def sort_filenames(filenames):
    return sorted(filenames, key=lambda x: extract_second_e_values(x))

# --- Combine CSVs by keyword ---
def df_combine_keyword(directory, key, Normalize=False):
    os.chdir(directory)
    df_filenames = glob.glob('*.csv')
    if "Master Analysis Sheet.csv" in df_filenames:
        df_filenames.remove("Master Analysis Sheet.csv")
    df_filenames = sort_filenames(df_filenames)
    df_filenames_key_list = [f for f in df_filenames if key in f or key.lower() in f]

    temp_dfs = []
    max_length = 0

    for n, df_name in enumerate(df_filenames_key_list):
        df = pd.read_csv(df_name, index_col=0)
        df.rename(columns={df.columns[0]: n}, inplace=True)

        if Normalize:
            first_val = df[n].iloc[0]
            df[n] = df[n] / first_val if first_val != 0 else df[n]

        meta = pd.DataFrame({
            n: [f"{directory}/{df_name}", key, n]
        }, index=['ID', 'ROI_Type', 'Genotype'])
        df_combined = pd.concat([meta, df], axis=0).reset_index(drop=True)
        temp_dfs.append(df_combined)
        max_length = max(max_length, len(df_combined))

    for i in range(len(temp_dfs)):
        if len(temp_dfs[i]) < max_length:
            temp_dfs[i] = temp_dfs[i].reindex(range(max_length))

    return pd.concat(temp_dfs, axis=1)

# --- Get average time vector from CSV index ---
def get_avg_time_vector(folderpath, keyword):
    df_filenames = glob.glob(os.path.join(folderpath, '*.csv'))
    df_filenames = [f for f in df_filenames if keyword in f or keyword.lower() in f]
    time_vectors = []

    for file in df_filenames:
        df = pd.read_csv(file, index_col=0)
        time_series = pd.to_numeric(df.index, errors='coerce')
        time_vectors.append(time_series.values)

    max_len = max(len(vec) for vec in time_vectors)
    time_matrix = np.full((len(time_vectors), max_len), np.nan)
    for i, vec in enumerate(time_vectors):
        time_matrix[i, :len(vec)] = vec
    return np.nanmean(time_matrix, axis=0)

# --- Process all ROI categories ---
categories = ['ROI_1', 'ROI_2', 'ROI_3', 'Nucleus', 'Control']

# Optional: create master sheet
dfs = [df_combine_keyword(folderpath, cat, Normalize=False) for cat in categories]
dfs_norm = [df_combine_keyword(folderpath, cat, Normalize=True) for cat in categories]
Master_Sheet = pd.concat(dfs + dfs_norm, axis=0)
Master_Sheet.to_csv(os.path.join(folderpath, 'Master Analysis Sheet.csv'), header=False)

# --- Color palette to match Illustrator style ---
color_palette = {
    'ROI_1':    ('#1f77b4', '#aec7e8'),  # Blue
    'ROI_2':    ('#ff7f0e', '#ffbb78'),  # Orange
    'ROI_3':    ('#2ca02c', '#98df8a'),  # Green
    'Nucleus':  ('#d62728', '#ff9896'),  # Red
    'Control':  ('#9467bd', '#c5b0d5'),  # Purple
    'Extra_1':  ('#8c564b', '#c49c94'),
    'Extra_2':  ('#e377c2', '#f7b6d2'),
    'Extra_3':  ('#7f7f7f', '#c7c7c7'),
}

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

for category in categories:
    df = df_combine_keyword(folderpath, category, Normalize=True)
    df_data = df.iloc[3:].astype(float)
    mean_values = df_data.mean(axis=1, skipna=True)
    std_values = df_data.std(axis=1, skipna=True)
    n = df_data.count(axis=1)
    ci_95 = 1.96 * (std_values / np.sqrt(n))

    avg_time = get_avg_time_vector(folderpath, category)
    avg_time = avg_time[:len(mean_values)]

    line_color, fill_color = color_palette.get(category, ('black', 'lightgray'))

    # --- Cutoff at 400 seconds ---
    time_mask = avg_time <= 600
    avg_time_cut = avg_time[time_mask]
    mean_cut = mean_values[time_mask]
    ci_cut = ci_95[time_mask]

    plt.plot(avg_time_cut, mean_cut, label=category, color=line_color, linewidth=2)
    plt.fill_between(avg_time_cut, mean_cut - ci_cut, mean_cut + ci_cut, color=fill_color, alpha=0.3)


plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Intensity (a.u.)")
plt.title("Time-course of Normalized ROI Intensities")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=False)
plt.tight_layout()

# --- Save ---
# --- Save before plt.show ---
plot_save_path = os.path.join(folderpath, 'XY_Plot_with_CI_Styled.png')
plt.savefig(plot_save_path, dpi=300)

# Save SVG (vector format)
plot_save_path_svg = os.path.join(folderpath, 'XY_Plot_with_CI_Styled.svg')
plt.savefig(plot_save_path_svg)

# Only now show the plot
plt.show()

print(f"✅ Plot saved as:\n  • PNG: {plot_save_path}\n  • SVG: {plot_save_path_svg}")
