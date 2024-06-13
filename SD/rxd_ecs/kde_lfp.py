import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from itertools import chain
import scipy.stats as st
from bokeh.plotting import figure, show
import plotly.offline as py
import plotly.graph_objects as go
from collections import OrderedDict

Y_OFFSET = 300
flatten = chain.from_iterable
varb = 1

const = 300


def load_cc_lfp(filepath):
    '''
        Load from folder with results
        Args:
    		filepath (str): path to the file
    	Returns:
    		dict: dictionary with keys and values
    '''
    lfp_cc = {}
    with h5py.File(filepath, 'r') as file:
        def traverse(group, prefix=""):
            for key in group.keys():
                item = group[key]
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Group):
                    traverse(item, path)
                elif isinstance(item, h5py.Dataset):
                    lfp_cc[path] = item[()]
        traverse(file)
    return lfp_cc


def plot_lfp(lfp, matplotlib=False, bokeh=False):
    '''
        plot lfp graph
        Args:
    		lfp: dictionary with keys and values
    		matplotlib (bool): choose lib
    		bokeh (bool): choose lib
    	Returns:
    		dict: dictionary with keys and values
        '''
    colors = ['black', 'navy', 'darkslateblue', 'indigo', 'purple', 'darkorchid', 'firebrick', 'indianred',
              'palevioletred', 'lightcoral', 'salmon', 'sandybrown', 'lightsalmon', 'orange', 'gold', 'yellow','yellowgreen']
    data = {}
    for key, value in lfp.items():
        new_key = int(key.split("/")[-1])
        data[new_key] = value

    sorted_keys = sorted(data.keys(), reverse=True)
    ordered_data = OrderedDict()
    for index, key in enumerate(sorted_keys):
        ordered_data[index] = data[key]

    if matplotlib:
        fig, ax = plt.subplots()

        for key, data in ordered_data.items():
            xticks = np.linspace(0, 50, len(data))
            ax.plot(xticks, data + key * const, lw=1.5, color=colors[key])

        ax.set_yticks(np.arange(16) * const)
        ax.set_yticklabels(np.arange(16))

        plt.xlabel('Time (ms)')
        plt.ylabel('Sensors № (mV)')
        plt.show()

    if bokeh:
        p = figure(x_axis_label='Time (ms)', y_axis_label='Sensors')
        # p.yaxis.major_label_text_font_size = '0pt'

        x_values = np.linspace(0, 50, num=2001)

        for key in ordered_data:
            # xticks = np.arange(len(data[key]))
            xticks = x_values
            y_offset = key * 3000
            # plot the curve
            p.line(xticks, np.array(ordered_data[key]) + y_offset, line_width=2, color=colors[key])
        show(p)

    return ordered_data


def find_extrema(array, condition):
    """
    	Advanced wrapper of numpy.argrelextrema
    	Args:
    		array (np.ndarray): data array
    		condition (np.ufunc): e.g. np.less (<), np.great_equal (>=) and etc.
    	Returns:
    		np.ndarray: indexes of extrema
    		np.ndarray: values of extrema
    """
    indexes = np.ndarray
    for i in array:
        indexes = argrelextrema(array, condition)[0]
        if len(indexes) == 0:
            return None, None

    values = array[indexes]
    diff_nearby_extrema = np.abs(np.diff(values, n=1))
    indexes = np.array([index for index, diff in zip(indexes, diff_nearby_extrema) if diff > 0] + [indexes[-1]])
    values = array[indexes]
    return indexes, values

def peak_finding(sensors_data, dstep, border_time, border_ampl, debug = False):
    """
    	Function for extrema (peaks) finding
    	Args:
    		channels_data (np.ndarray): 2D array (channel, channel data)
    		dstep (float): data step size
    		border_time (list): the time border [min, max]
    		border_ampl (list): the amplitude border [min, max]
    		debug (bool): debug mode (plotting)
    	Returns:
    		list: peak times, amplitudes and channels
    """
    layers_num, _ = sensors_data.shape

    peaks_time= [[] for _ in range(layers_num)]
    peaks_ampl = [[] for _ in range(layers_num)]
    peaks_chan = [[] for _ in range(layers_num)]

    p = figure(x_axis_label='Time (ms)', y_axis_label='Sensors')
    p.yaxis.major_label_text_font_size = '0pt'

    fig, ax = plt.subplots()

    for index, channel in enumerate(sensors_data):
        # combine slices into one myogram
        e_max_inds, e_max_vals = find_extrema(channel, np.greater)
        e_min_inds, e_min_vals = find_extrema(channel, np.less)

        # start pairing extrema from maxima
        offset = slice(1, None) if e_min_inds[0] < e_max_inds[0] else slice(None)
        comb = list(zip(e_max_inds, e_min_inds[offset]))

        # create list for debugging (plot max values of peaks)
        max_value_peaks = []
        # process each extrema pair
        for max_index, min_index in comb:
            max_value = e_max_vals[e_max_inds == max_index][0]
            min_value = e_min_vals[e_min_inds == min_index][0]
            dT = abs(max_index - min_index) * dstep
            dA = abs(max_value - min_value)
            # check the difference between maxima and minima
            if (border_time[0] <= dT <= border_time[1]) and border_ampl[0] <= dA <= border_ampl[1]:
                peaks_time[index].append(max_index)
                peaks_ampl[index].append(dA)
                peaks_chan[index].append(index)
                max_value_peaks.append(max_value)

        if debug:
            xticks = np.arange(len(channel)) * dstep
            y_offset = index * Y_OFFSET
            # plot the curve
            plt.plot(xticks, channel + y_offset, color='k')
            # plot the extrema
            plt.plot(e_max_inds * dstep, e_max_vals + y_offset, '.', color='r')
            plt.plot(e_min_inds * dstep, e_min_vals + y_offset, '.', color='b')
            # plot the peaks
            x = np.asarray(peaks_time[index]) * dstep
            y = np.asarray(max_value_peaks) + np.asarray(peaks_chan[index]) * Y_OFFSET
            plt.plot(x, y, '.', color='g', ms=5)

        # if debug:
        #     xticks = np.arange(len(channel)) * dstep
        #     y_offset = index * Y_OFFSET
        #     # plot the curve
        #     p.line(xticks, channel + y_offset, color=colors[0])
        #     # plot the extrema
        #     p.circle(e_max_inds * dstep, e_max_vals + y_offset, color=colors[1])
        #     p.circle(e_min_inds * dstep, e_min_vals + y_offset, color=colors[2])
        #     # plot the peaks
        #     x = np.asarray(peaks_time[index]) * dstep
        #     y = np.asarray(max_value_peaks) + np.asarray(peaks_chan[index]) * Y_OFFSET
        #     p.circle(x, y, color=colors[3])

    if debug:
        ax.set_yticks(np.arange(16) * Y_OFFSET)
        ax.set_yticklabels(np.arange(16))

        plt.xlabel('Time (ms)')
        plt.ylabel('Sensors № (mV)')
        plt.show()
        # show(p)

    return peaks_time, peaks_ampl, peaks_chan

def plot_3D_density(X, Y, xmin, xmax, ymin, ymax, factor=8, filepath=None):
    """
    	Plots the 3D density graphics
    	Args:
    		X (np.ndarray): flatten 1D array
    		Y (np.ndarray): flatten 1D array
    		xmin (float or int): minimal X data value
    		xmax (float or int): maximal X data value
    		ymin (float or int): minimal Y data value
    		ymax (float or int): maximal Y data value
    		factor (float or int): gridsize factor
    		filepath (str): filepath for .html saving
    """
    if filepath is None:
        return
    # form a mesh grid
    gridsize_x, gridsize_y = factor * xmax, factor * ymax
    xmesh, ymesh = np.meshgrid(np.linspace(xmin, xmax, gridsize_x),
                               np.linspace(ymin, ymax, gridsize_y))
    xmesh = xmesh.T
    ymesh = ymesh.T
    # re-present grid in 1D and pair them as (x1, y1 ...)
    positions = np.vstack([xmesh.ravel(), ymesh.ravel()])
    values = np.vstack((X, Y))
    # use a Gaussian KDE
    a = st.gaussian_kde(values)(positions).T

    # with open('filepath', 'w') as f:
    #     for line in a:
    #         f.write(str(line))
    #         f.write('\n')
    #         # print(1)

    # re-present grid back to 2D
    z = np.reshape(a, xmesh.shape)

    z_contours = {"show": True,
                  "start": np.min(z) - 0.00001,
                  "end": np.max(z) + 0.00001,
                  "size": (np.max(z) - np.min(z)) / 16,
                  'width': 1,
                  "color": "gray"}
    surface = go.Surface(x=xmesh, y=ymesh, z=z, contours=dict(z=z_contours), opacity=1)

    # plot the 3D
    fig = go.Figure(data=surface)
    # change a camera view and etc
    fig.update_layout(title=f'Sim rat 1', width=1000, height=800, autosize=False,
                      scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.25, y=-1.25, z=1.25)),
                      scene=dict(xaxis=dict(title_text="Time, ms",
                                            titlefont=dict(size=30),
                                            ticktext=list(range(26))),
                                 yaxis=dict(title_text="Sensor №",
                                            titlefont=dict(size=30),
                                            tickvals=list(range(ymax + 1)),
                                            ticktext=list(range(1, ymax + 2))),
                                 zaxis=dict(title_text="Density"),
                                 aspectratio={"x": 1, "y": 1, "z": 0.5}))

    py.plot(fig, validate=False, filename=f"{filepath}/test.html", auto_open=True)


def main():
    path = ''
    lfp_cc = load_cc_lfp(path)
    lfp = plot_lfp(lfp_cc, matplotlib=True)

    file_folder = ''

    border_time = [.0, 250]
    border_ampl = [-100, np.inf]
    dstep = 0.025

    lfp_data = np.array(list(lfp.values()))
    peaks_time, peaks_ampl, peaks_chan = peak_finding(lfp_data, dstep, border_time, border_ampl, debug=True)

    x_data = np.array(list(flatten(peaks_time))) * 0.025
    y_data = np.array(list(flatten(peaks_chan)))
    z_data = np.array(list(flatten(peaks_ampl)))

    plot_3D_density(x_data, y_data, xmin=0, xmax=60, ymin=0, ymax=17, filepath=file_folder)


if __name__ == "__main__":
    main()