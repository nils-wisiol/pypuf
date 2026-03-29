import numpy as np
import pandas as pd
import glob
import os

# ==============================
# 1. FIND ALL PUF CSV FILES
# ==============================

# Search for all *_master_accuracy.csv files in current directory only
csv_pattern = "*_master_accuracy.csv"
csv_files = glob.glob(csv_pattern)

print(f"Found {len(csv_files)} CSV files in current directory:")
for f in sorted(csv_files):
    print(f"  - {f}")

# ==============================
# 2. PROCESS EACH CSV AND COLLECT STATISTICS
# ==============================

summary_stats = []

for csv_file in sorted(csv_files):
    df = pd.read_csv(csv_file)
    
    # Extract PUF type from filename (remove _master_accuracy.csv)
    filename = os.path.basename(csv_file)
    puf_name = filename.replace("_master_accuracy.csv", "")
    
    # Get accuracy column
    if "Accuracy" in df.columns:
        accuracies = df["Accuracy"].values
        
        # Calculate statistics
        stats = {
            "PUF_Type": puf_name,
            "Max_Accuracy": np.max(accuracies),
            "Min_Accuracy": np.min(accuracies),
            "Average_Accuracy": np.mean(accuracies),
            "Std_Accuracy": np.std(accuracies),
            "Num_Points": len(accuracies)
        }
        summary_stats.append(stats)
        
        print(f"\n{puf_name}:")
        print(f"  Max:     {stats['Max_Accuracy']:.4f}")
        print(f"  Min:     {stats['Min_Accuracy']:.4f}")
        print(f"  Average: {stats['Average_Accuracy']:.4f}")
        print(f"  Std Dev: {stats['Std_Accuracy']:.4f}")
    else:
        print(f"\nWarning: 'Accuracy' column not found in {filename}")

# ==============================
# 3. CREATE SUMMARY DATAFRAME
# ==============================

if summary_stats:
    df_summary = pd.DataFrame(summary_stats)
    
    # Save to CSV in current directory
    output_file = "MLP_PUF_Statistics_Summary.csv"
    df_summary.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Summary statistics saved to: {output_file}")
    print(f"{'='*60}\n")
    print(df_summary.to_string(index=False))
else:
    print("No CSV files found or processed.")