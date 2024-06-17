import numpy as np
from scipy.stats import ks_2samp
import statistics as st
import os

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f]
    return np.array(data)

def calculate_iou(arr1, arr2):
    startA = min(arr1)
    endA = max(arr1)
    startB = min(arr2)
    endB = max(arr2)

    overlap = max(0, min(endA, endB) - max(startA, startB))

    boxAArea = endA - startA
    boxBArea = endB - startB

    iou = overlap / float(boxAArea + boxBArea - overlap)

    return iou

def process_data(cc_file, mat_file, output_file):
    lfp_cc = load_txt(cc_file)
    lfp_mat = load_txt(mat_file)

    iou = calculate_iou(lfp_cc, lfp_mat)
    # print(f"IoU между массивом модели {os.path.basename(cc_file)} и массивом модели {os.path.basename(mat_file)}: {iou}")

    # Kolmogorov-Smirnov test
    subsample_size = min(10000, len(lfp_cc), len(lfp_mat))
    arr1 = np.random.choice(lfp_cc, size=subsample_size, replace=False)
    arr2 = np.random.choice(lfp_mat, size=subsample_size, replace=False)
    result = ks_2samp(arr1, arr2, method='asymp')

    # Comparison of averages
    mean_cc = st.mean(lfp_cc)
    mean_bio = st.mean(lfp_mat)
    # print(f"{result}")
    # print(f"Среднее значение в {os.path.basename(cc_file)}: {mean_cc}")
    # print(f"Среднее значение в {os.path.basename(mat_file)}: {mean_bio}")

    # Запись результатов в файл
    with open(output_file, "a") as f:
        f.write(f"Файл 1: {os.path.basename(cc_file)}\n")
        f.write(f"Файл 2: {os.path.basename(mat_file)}\n")
        f.write(f"IoU: {iou}\n")
        f.write(f"Kolmogorov-Smirnov test: {result}\n")
        f.write(f"Среднее значение в {os.path.basename(cc_file)}: {mean_cc}\n")
        f.write(f"Среднее значение в {os.path.basename(mat_file)}: {mean_bio}\n\n")

def main():
    cc_folder = ""
    mat_folder = ""
    output_file = ""

    cc_files = [os.path.join(cc_folder, f) for f in os.listdir(cc_folder) if os.path.isfile(os.path.join(cc_folder, f))]
    mat_files = [os.path.join(mat_folder, f) for f in os.listdir(mat_folder) if os.path.isfile(os.path.join(mat_folder, f))]

    with open(output_file, "w") as f:
        f.write("")

    for cc_file in cc_files:
        for mat_file in mat_files:
            process_data(cc_file, mat_file, output_file)

if __name__ == "__main__":
    main()