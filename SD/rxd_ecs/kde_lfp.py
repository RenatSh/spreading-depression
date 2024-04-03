import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema
from itertools import chain
import scipy.stats as st
from bokeh.plotting import figure, show
import plotly.offline as py
import plotly.graph_objects as go

Y_OFFSET = 1000
flatten = chain.from_iterable
varb = 1


def load_cc_lfp(filepath):
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

def plotting_lfp(lfps):
    yx=1
    fig, ax = plt.subplots()
    keys = list((lfps.keys()))
    keys.sort(reverse=True)
    legend = []
    for index, key in enumerate(keys):
        plt.plot(np.array(lfps[key]) + yx*1e-11)
        yx+=1
        legend.append(index)
    plt.legend(legend)
    plt.show()
    print(1)

def plot_lfp_bokeh(lfp):
    colors = ['black', 'red', 'green', 'blue', 'indigo', 'crimson', 'orange', 'gold', 'gray','maroon', 'navy', 'purple', 'olive', 'cyan', 'brown', 'lime']
    keys = list(lfp.keys())
    keys.sort(reverse=True)
    yx = 1

    p = figure(x_axis_label='Ms', y_axis_label='Sensors')
    p.yaxis.major_label_text_font_size = '0pt'
    x_values = np.linspace(0, 50, num=2001)

    for i, key in enumerate(keys):
        y = np.array(lfp[key]) + yx * 1600
        x = x_values
        # x = np.arange(len(lfp[key]))
        # x = np.arange(len(lfp[key]))
        p.line(x, y, line_width=2, color=colors[i])
        yx+=1
    show(p)

    p.legend.location = 'top_left'

    # show(p)
    print(1)

def find_extrema(array, condition):
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

    # layers_num = len(sensors_data)
    layers_num, _ = sensors_data.shape

    varb = 1

    peaks_time= [[] for _ in range(layers_num)]
    peaks_ampl = [[] for _ in range(layers_num)]
    peaks_chan = [[] for _ in range(layers_num)]

    colors = ['black', 'red', 'blue', 'green']

    p = figure(x_axis_label='Ms', y_axis_label='Sensors')
    p.yaxis.major_label_text_font_size = '0pt'

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

        # if debug:
        #     xticks = np.arange(len(channel)) * dstep
        #     y_offset = index * 1e-9
        #     # plot the curve
        #     plt.plot(xticks, channel + y_offset, color='k')
        #     # plot the extrema
        #     plt.plot(e_max_inds * dstep, e_max_vals + y_offset, '.', color='r')
        #     plt.plot(e_min_inds * dstep, e_min_vals + y_offset, '.', color='b')
        #     # plot the peaks
        #     x = np.asarray(peaks_time[index]) * dstep
        #     y = np.asarray(max_value_peaks) + np.asarray(peaks_chan[index]) * Y_OFFSET
        #     plt.plot(x, y, '.', color='g', ms=20)

        if debug:
            xticks = np.arange(len(channel)) * dstep
            y_offset = index * Y_OFFSET
            # plot the curve
            p.line(xticks, channel + y_offset, color=colors[0])
            # plot the extrema
            p.circle(e_max_inds * dstep, e_max_vals + y_offset, color=colors[1])
            p.circle(e_min_inds * dstep, e_min_vals + y_offset, color=colors[2])
            # plot the peaks
            x = np.asarray(peaks_time[index]) * dstep
            y = np.asarray(max_value_peaks) + np.asarray(peaks_chan[index]) * Y_OFFSET
            p.circle(x, y, color=colors[3])

    if debug:
        # plt.show()
        show(p)

    return peaks_time, peaks_ampl, peaks_chan

def plot_3D_density(X, Y, xmin, xmax, ymin, ymax, factor=8, filepath=None):

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

    # with open('', 'w') as f:
    #     for line in a:
    #         f.write(str(line))
    #         f.write('\n')
            # print(1)

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
    fig.update_layout(title=f'test title', width=1000, height=800, autosize=False,
                      scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-1.25, y=-1.25, z=1.25)),
                      scene=dict(xaxis=dict(title_text="Time, ms",
                                            titlefont=dict(size=30),
                                            ticktext=list(range(26))),
                                 yaxis=dict(title_text="Channel №",
                                            titlefont=dict(size=30),
                                            tickvals=list(range(ymax + 1)),
                                            ticktext=list(range(1, ymax + 2))),
                                 aspectratio={"x": 1, "y": 1, "z": 0.5}))

    py.plot(fig, validate=False, filename=f"{filepath}/12.html", auto_open=True)


def main():
    path = ''
    lfp_cc = load_cc_lfp(path)

    # plotting_lfp(lfp_cc)
    plot_lfp_bokeh(lfp_cc)

    file_folder = 'D:/results/grafs'

    border_time = [.0, 250]
    border_ampl = [-100, np.inf]
    dstep = 0.025

    lfp_data = np.array(list(lfp_cc.values()))
    peaks_time, peaks_ampl, peaks_chan = peak_finding(lfp_data, dstep, border_time, border_ampl, debug=True)

    x_data = np.array(list(flatten(peaks_time))) * 0.025
    y_data = np.array(list(flatten(peaks_chan)))
    z_data = np.array(list(flatten(peaks_ampl)))

    plot_3D_density(x_data, y_data, xmin=0, xmax=60, ymin=0, ymax=17, filepath=file_folder)

    # calc_kde(lfp_data)


if __name__ == "__main__":
    main()