import os
import json
import numpy as np

directory = "/home/il016930/InjectAgent_to_post/results/thoughts_analysis_path_ablation_new"

after_hesitation_asrs = []
after_safe_reflector_asrs = []

for filename in os.listdir(directory):
    if filename.startswith("asr_summary_") and filename.endswith(".json.json"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            after_hesitation_asrs.append(data["after_hesitation_asr"])
            after_safe_reflector_asrs.append(data["after_safe_reflector_asr"])

after_hesitation_asrs = np.array(after_hesitation_asrs)
after_safe_reflector_asrs = np.array(after_safe_reflector_asrs)
differences = after_hesitation_asrs - after_safe_reflector_asrs

mean_after_hesitation = np.mean(after_hesitation_asrs)
mean_after_safe = np.mean(after_safe_reflector_asrs)
std_after_hesitation = np.std(after_hesitation_asrs, ddof=1)
std_after_safe = np.std(after_safe_reflector_asrs, ddof=1)
mean_difference = np.mean(differences)
std_difference = np.std(differences, ddof=1)

print("Statistics:")
print(f"Mean After Hesitation ASR: {mean_after_hesitation:.2f}")
print(f"Mean After Safe Reflector ASR: {mean_after_safe:.2f}")
print(f"STD After Hesitation ASR: {std_after_hesitation:.2f}")
print(f"STD After Safe Reflector ASR: {std_after_safe:.2f}")
print(f"Mean Difference: {mean_difference:.2f}")
print(f"STD of Difference: {std_difference:.2f}")
