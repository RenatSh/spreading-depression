import logging
import os
import h5py
import scipy.io
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objects as go
from bokeh.plotting import figure, show, output_notebook, output_file
from scipy.ndimage import gaussian_filter


from itertools import chain
from scipy.signal import argrelextrema

flatten = chain.from_iterable
Y_OFFSET = 1000

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
log = logging.getLogger()


def load_data(filepath):
    """
	Load data from the .mat format
	Args:
		filepath (str): path to the file
	Returns:
		dict: dictionary with variable names as keys, and loaded matrices as values
	"""
    mat = scipy.io.loadmat(filepath)
    log.info(f"File ({filepath}) loaded")
    return mat



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
    # get indexes of extrema
    indexes = argrelextrema(array, condition)[0]
    # in case where data line is horisontal and doesn't have any extrema -- return None
    if len(indexes) == 0:
        return None, None
    # get values based on found indexes
    values = array[indexes]
    # calc the difference between nearby extrema values
    diff_nearby_extrema = np.abs(np.diff(values, n=1))
    # form indexes where no twin extrema (the case when data line is horisontal and have two extrema on borders)
    indexes = np.array([index for index, diff in zip(indexes, diff_nearby_extrema) if diff > 0] + [indexes[-1]])
    # get values based on filtered indexes
    values = array[indexes]

    return indexes, values


def find_peaks(channels_data, dstep, border_time, border_ampl, debug=False):
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
    channels_num, _ = channels_data.shape
    # generate 2D list by channels number
    peaks_time = [[] for _ in range(channels_num)]
    peaks_ampl = [[] for _ in range(channels_num)]
    peaks_chan = [[] for _ in range(channels_num)]

    colors = ['black', 'red', 'blue', 'green']
    p = figure(x_axis_label='Ms', y_axis_label='Sensors')
    p.yaxis.major_label_text_font_size = '0pt'

    fig, ax = plt.subplots()

    for index, channel in enumerate(channels_data):
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
        #

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
            plt.plot(x, y, '.', color='g', ms=10)

    if debug:
        ax.set_yticks(np.arange(15)*1000)
        ax.set_yticklabels(np.arange(15))

        plt.xlabel('Time (ms)')
        plt.ylabel('Sensors № (V)')
        # show(p)
        plt.show()

    return peaks_time, peaks_ampl, peaks_chan


def smooth_data(channels, box_pts):
    """
	Smoothing a curves
	Args:
		channels (np.ndarray): 2D array, y-axis data
		box_pts (int):
	Returns:
		np.ndarray: smoothed aata
	"""
    box = np.ones(box_pts) / box_pts
    for channel in channels:
        channel[:] = np.convolve(channel, box, mode='same')

    return channels

def filter_data(channels, dstep, bokeh= False, matplotlib=False):
    for channel in channels:
        channel[:] = gaussian_filter(channel, sigma=1)

    p = figure(x_axis_label='Time (ms)', y_axis_label='Sensors №')
    p.yaxis.major_label_text_font_size = '0pt'
    colors = ['black', 'navy', 'darkslateblue', 'indigo', 'purple', 'darkorchid', 'firebrick', 'indianred',
              'palevioletred', 'lightcoral', 'salmon', 'sandybrown', 'lightsalmon', 'orange', 'gold', 'yellow',
              'yellowgreen']
    a, b = channels.shape

    if matplotlib:
        fig, ax = plt.subplots()
        for index, channel in enumerate(channels):
            xticks = np.linspace(0, 50, b)
            ax.plot(xticks, channel + index * Y_OFFSET, lw=1.5, color=colors[index])
        ax.set_yticks(np.arange(15) * 1000)
        ax.set_yticklabels(np.arange(15))

        plt.xlabel('Time (ms)')
        plt.ylabel('Sensors № (V)')

        plt.show()

    if bokeh:
        for index, channel in enumerate(channels):
            xticks = np.arange(len(channel)) * dstep
            y_offset = index * Y_OFFSET
            # plot the curve
            p.line(xticks, channel + y_offset, line_width=2, color=colors[index])
        show(p)
    return channels


def get_channels(data, name, channels=None, dstep=0.025, debug=False):
    """
	Get the data from the dict mat file
	Args:
		data (dict): dict of the mat file
		name (str): name of the channels?
		channels (list): a range list of channels for extracting
		dstep (float): data step size
		debug (bool): debug mode (plotting)
	Returns:
		np.ndarray: 2D array (channel, values)
	"""
    extracted = data.get(name, None)
    if extracted is None:
        raise KeyError(f"The key '{name}' does not exist!")

    log.info(f"data shape {name} {extracted.shape}")

    # re-order to 1. experiment; 2. channel; 3. channel data
    extracted = np.moveaxis(extracted, [0, 1, 2], [2, 1, 0])
    exp_num, cha_num, val_num = extracted.shape

    # FIXME get the channels of the 30th experiment?!
    channels = slice(*channels)
    channels_data = extracted[30, channels, :]

    p = figure(title='График lfp', x_axis_label='Ms', y_axis_label='Sensors')
    p.yaxis.major_label_text_font_size = '0pt'

    if debug:
        exp_time = val_num * dstep
        # fig, ax = plt.subplots(figsize=(30, 7))
        # ax.set_title("Channels")
        for index, channel in enumerate(channels_data):
            xticks = np.linspace(0, exp_time, val_num)
            p.line(xticks, channel + index * Y_OFFSET)
        show(p)

    # if debug:
    #     exp_time = val_num * dstep
    #     fig, ax = plt.subplots(figsize=(30, 7))
    #     ax.set_title("Channels")
    #     for index, channel in enumerate(channels_data):
    #         xticks = np.linspace(0, exp_time, val_num)
    #         ax.plot(xticks, channel + index * Y_OFFSET, lw=1)
    #     plt.show()

    log.info("Shape of the channels")
    log.info(f"Shape {channels_data.shape}")

    return channels_data


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

    with open('', 'w') as f:
        for line in a:
            f.write(str(line))
            f.write('\n')

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
    fig.update_layout(title=f'KDE rat №5', width=1000, height=800, autosize=False,
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
    file_prefix = ""
    filename = ''
    filepath = f"{file_prefix}/{filename}"
    # file_prefix_e = ""
    # filename_e = ""
    # filepath_e = f"{file_prefix_e}/{filename_e}"

    dstep = 0.025
    channels = [0, 15]
    border_time = [0.1, 3]
    border_ampl = [100, np.inf]

    data = load_data(filepath)
    channel_data = get_channels(data, channels=channels, name='lfp', debug=False)
    channel_data = smooth_data(channel_data, 20)
    channel_data = filter_data(channel_data, dstep, matplotlib=True)

    peaks_time, peaks_ampl, peaks_chan = find_peaks(channel_data, dstep, border_time, border_ampl, debug=True)

    # choose the data
    x_data = np.array(list(flatten(peaks_time))) * dstep
    y_data = np.array(list(flatten(peaks_chan)))
    z_data = np.array(list(flatten(peaks_ampl)))

    plot_3D_density(x_data, y_data, xmin=0, xmax=60, ymin=0, ymax=15, filepath=file_prefix)


if __name__ == "__main__":
    main()
