import numpy as np
from scipy.stats import kstwo, ks_2samp, kstest
import statistics as st

def load_txt(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(float(line.split("\n")[0]))
    return data

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


def main():
    file = ""
    matfile = ""

    lfp_cc = load_txt(file)
    lfp_mat = load_txt(matfile)


    iou = calculate_iou(lfp_cc, lfp_mat)
    print(f"IoU между массивом модели 1 и массивом модели 2: {iou}")

    #Kolmogorov-Smirnov test

    subsample_size = 10000
    arr1 = np.random.choice(lfp_cc, size=subsample_size, replace=False)
    arr2 = np.random.choice(lfp_mat, size=subsample_size, replace=False)

    result = ks_2samp(arr1, arr2, method='asymp')

    #Comparison of averages
    mean_cc = st.mean(lfp_cc)
    mean_bio = st.mean(lfp_mat)
    print(f"{result}")
    print("Среднее значение в lfp_cc " + str(mean_cc))
    print("Среднее значение в lfp_mat " + str(mean_bio))


if __name__ == "__main__":
    main()