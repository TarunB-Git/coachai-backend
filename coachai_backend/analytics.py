import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_differences_to_csv(differences, joint_names, output_csv="differences.csv"):
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + joint_names)
            for i, frame_diff in enumerate(differences):
                row = [i] + [f"{d:.4f}" if d is not None else "" for d in frame_diff]
                writer.writerow(row)
        print(f"Exported differences to {output_csv}")
    except Exception as e:
        print("Error exporting CSV:", e)


def plot_average_errors(avg_diffs, joint_names, output_path="avg_errors.png"):
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(avg_diffs)), avg_diffs, color="skyblue")
        plt.xticks(range(len(avg_diffs)), joint_names, rotation=90)
        plt.xlabel("Joint")
        plt.ylabel("Average Error")
        plt.title("Average Error per Joint")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved average error chart to {output_path}")
    except Exception as e:
        print("Error plotting average errors:", e)


def plot_joint_errors_over_time(differences, joint_indices, joint_names, output_prefix="joint_errors"):
    num_frames = len(differences)
    for idx in joint_indices:
        errors = [frame[idx] if frame[idx] is not None else np.nan for frame in differences]
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_frames), errors, marker="o", linestyle="-")
            plt.xlabel("Frame Index")
            plt.ylabel("Error")
            plt.title(f"Error over Time: {joint_names[idx]}")
            plt.tight_layout()
            out_file = f"{output_prefix}_{joint_names[idx]}.png"
            plt.savefig(out_file)
            plt.close()
            print(f"Saved time-series for {joint_names[idx]} to {out_file}")
        except Exception as e:
            print(f"Error plotting errors for joint {joint_names[idx]}:", e)
