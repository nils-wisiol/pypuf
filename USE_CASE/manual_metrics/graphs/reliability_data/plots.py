import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_separated_minima_with_skip():
    # List all CSV files in the directory
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    summary_data = []

    for file in all_files:
        # Avoid processing specific exclusion files or previous reports
        if any(exclude in file for exclude in ["XORBistableRingPUF", "summary", "report", "comparison"]):
            continue
            
        try:
            df = pd.read_csv(file)
            
            # Verify required columns exist before filtering
            if not all(col in df.columns for col in ['Reliability', 'Temperature', 'Vdd']):
                continue

            # --- GLOBAL FILTER: Skip the nominal point (T=20, Vdd=1.35) ---
            # This ensures artifacts at this specific coordinate don't affect any result
            df = df[~(np.isclose(df['Temperature'], 20) & np.isclose(df['Vdd'], 1.35))]
            
            if df.empty:
                continue

            puf_name = file.split('_')[0]
            
            # Find the row containing the absolute minimum reliability in the filtered data
            min_idx = df['Reliability'].idxmin()
            min_row = df.loc[min_idx]
            
            # Classify based on your class naming convention
            if file.endswith('_performance_data_sweep.csv'):
                sweep_type = 'Temperature Stress (Constant Vdd)'
            else:
                sweep_type = 'Voltage Stress (Constant T)'
            
            summary_data.append({
                'PUF Architecture': puf_name,
                'Stress Type': sweep_type,
                'Min Reliability': min_row['Reliability'],
                'At Temp (C)': min_row['Temperature'],
                'At Vdd (V)': min_row['Vdd']
            })
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Create summary and sort for clear architectural comparison
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by=['PUF Architecture', 'Stress Type'])
    
    # Save results to a summary report
    output_csv = 'puf_minima_no_nominal_report.csv'
    summary_df.to_csv(output_csv, index=False)
    
    # --- Plotting the Results ---
    pivot_df = summary_df.pivot(index='PUF Architecture', columns='Stress Type', values='Min Reliability')
    ax = pivot_df.plot(kind='bar', figsize=(12, 7), color=['#3498db', '#e74c3c'], width=0.8)
    
    plt.title('Worst-Case Reliability ($R_{min}$) ', fontsize=14)
    plt.ylabel('Minimum Reliability ($R_{min}$)')
    plt.xticks(rotation=45, ha='right')
    
    # Set Y-axis floor based on the lowest reliability found
    global_min = summary_df['Min Reliability'].min()
    plt.ylim(max(0, global_min - 0.05), 1.02)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("Worst-case Reliability Report :")
    print(summary_df[['PUF Architecture', 'Stress Type', 'Min Reliability', 'At Temp (C)', 'At Vdd (V)']])

if __name__ == "__main__":
    extract_separated_minima_with_skip()