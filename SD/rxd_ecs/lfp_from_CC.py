import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import csv
import  math


hdf5_files='D:/results/ep1'
coordinates_file='D:/results/ep1/cord.csv'


def load_voltages(path):
    '''
    Load from folder with results
    Do not read files with epilepsy
    Args:
		filepath (str): path to the file
	Returns:
		dict: dictionary with keys and values
    '''
    vol_group = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.startswith("voltage_extracellular") and entry.name.endswith(".hdf5") and entry.is_file():
                file_name = os.path.splitext(entry.name)[0]
                file_name=file_name.split('_')[3]
                with h5py.File(entry.path, 'r') as f:
                    vol_group[file_name] = np.array(f[list(f.keys())[0][:]])
    return vol_group

def load_coordinates(path):
    '''
    Load from folder with coords
    Args:
		filepath (str): path to the file
	Returns:
		dict: dictionary with keys and values (coordinates)
    '''
    coordinates = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            coordinates.append(row)
    coordinates.sort()
    return coordinates


def calc_lfp(voltages, coords):
    '''
    Calculation ∑jIj/|r−rj| (extracellular potential divided by the square of the distance)
    Args:
        voltages: dictionary,
        coords: dictionary
	Returns:
		fraction_dict: dictionary with keys(№ sensor) and values (Lfp)

    Sensors на L23: 1, 2, 3
    Sensors на L4: 4, 5, 6
    Sensors на L5: 7, 8, 9
    Sensors на L56: 10, 11
    Sensors на L6: 12, 13, 14
    Sensors на tf: 15, 16
    '''
    sensors = [[1, 11, 18, -700], [2, 19, 27, -650], [3, 28, 36, -490], [4, 37, 45, -400],
               [5, 46, 54, -300], [6, 55, 63, -250], [7, 64, 72, -100], [8, 73, 81, 50],
               [9, 82, 90, 100], [10, 91, 99, 250], [11, 100, 108, 330], [12, 109, 117, 400],
               [13, 118, 126, 550], [14, 127, 135, 700], [15, 136, 144, 1120], [16, 145, 153, 1270]]
    fraction_dict = {}

    for sensor in sensors:
        arr = np.zeros(len(voltages[coords[0][0]]))
        j = 0
        sensor_number = sensor[0]
        sX = np.random.uniform(0, 189)
        sY = np.random.uniform(0, 189)
        sZ = np.random.uniform(-700, 1270)

        for coord in coords:
            x = float(coord[1])
            y = float(coord[2])
            z = float(coord[3])
            square_of_distance = abs((sX - x) ** 2 + (sY - y) ** 2 + (sZ - z) ** 2)

            for k, i in enumerate(voltages.get(coord[0])):
                fraq = i / square_of_distance
                arr[k] += fraq

        fraction_dict[str(sensor_number)] = np.array(arr)

    return fraction_dict


def calculation_simple_LFP(fraction):
    '''
    Constant multiplied by fraction
    Args:
		fraction_dict: dictionary with keys(№ sensor) and values (Lfp)
	Returns:
	    results: dictionary with keys(№ sensor) and values (Lfp)
    '''
    result = {}
    const = 1 / (4 * math.pi)
    for key, value in fraction.items():
        # lfp = const * value
        lfp = value * 1e13
        result[key] = np.array(lfp)
    return result



def main():
    voltages = load_voltages(hdf5_files)
    coordinates = load_coordinates(coordinates_file)
    fraction = calc_lfp(voltages, coordinates)
    lfp = calculation_simple_LFP(fraction)

    #save lpf in hdf5 file
    with h5py.File('D:/results/lfp/ts50e', 'w') as hf:
        group = hf.create_group('lfp')
        for key, value in lfp.items():
            group[key] = value


if __name__ == "__main__":
    main()