import os
import sys
import numpy as np
import pandas as pan
import matplotlib.pyplot as plt
import scipy.signal as sps

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import json

# import scienceplots
import data_file_reader

# import plotly.express as px

# (plt.style.use("science"))
np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=1000000000000, linewidth=1000000000000)

folder_path = r"D:\-SFU Physics Lab Work\HIM Room\Guneet\2024-07-11\RGA\Scan 3 - Increased Points - 2024_07_11\Analog-20240711-123603-709.rgadata"

time_stamps, spectra, total_pressures, json_file, number_of_cycles = data_file_reader.read_file_data(folder_path)

json_dictionary = json.loads(json_file)

print(number_of_cycles)

# print(json_dictionary)
print(total_pressures)
print(sum(time_stamps) / (len(time_stamps)))

# json_split = json_file.replace(",", "").split("\n")
points_per_amu = json_dictionary["cfgs"][0]["pointsPerAmu"]  # int((json_split[30])[28:])
scan_rate = json_dictionary["cfgs"][0]["scanRate"]  # float((json_split[31])[24:])
start_mass = json_dictionary["cfgs"][0]["startMass"]  # int((json_split[32])[25:])
stop_mass = json_dictionary["cfgs"][0]["stopMass"]  # int((json_split[33])[24:])

print(points_per_amu, scan_rate, start_mass, stop_mass)


def import_to_array():

    global rga_scan_data_array
    global amu_layer_vector
    global MASS_AMU_LAYER_INDEX
    MASS_AMU_LAYER_INDEX = 0
    global INTENSITY_TORR_LAYER_INDEX
    INTENSITY_TORR_LAYER_INDEX = 1
    global TIME_SECONDS_LAYER_INDEX
    TIME_SECONDS_LAYER_INDEX = 2
    # The array has a height of 3 to accomadate the AMU value, the time, and the pressure
    LAYER_DIMENSIONS = 3

    number_of_amu_points = ((stop_mass - start_mass) * points_per_amu) + 1
    # array initialized for the right amount of space
    rga_scan_data_array = np.ones((number_of_amu_points, number_of_cycles, LAYER_DIMENSIONS))
    # its maybe more convinent to have the AMU position bundled in with the main array, may change if any preformance/ease of use issues occour
    amu_layer_vector = np.linspace(start_mass, stop_mass, number_of_amu_points)
    amu_layer_array = amu_layer_vector[:, np.newaxis]
    amu_layer_array = np.repeat(amu_layer_array, number_of_cycles, 1)
    rga_scan_data_array[:, :, MASS_AMU_LAYER_INDEX] = amu_layer_array

    print("test", np.shape(rga_scan_data_array))

    difference = pressure_delta_calc(time_stamps, scan_rate, stop_mass)

    for i in range(number_of_cycles):

        rga_scan_data_array[:, i, INTENSITY_TORR_LAYER_INDEX] = spectra[i]

        for n in range(number_of_amu_points):
            rga_scan_data_array[n, i, TIME_SECONDS_LAYER_INDEX] = (
                n * 200_000 / number_of_amu_points + sum(difference[:i]) + 200_000 * i
            )

    rga_scan_data_array[:, :, MASS_AMU_LAYER_INDEX] = amu_layer_array

def pressure_delta_calc(timings, rate, max):

    # timings = [x/1000 for x in timings]
    print("Real timings:", timings)

    # guess = []
    # for i in range(len(timings)-1):
    #     guess.append(timings[] + i * 208)

    real_delta = []
    for i in range(len(timings) - 1):
        real_delta.append(timings[i + 1] - timings[i])

    differences = [x - (int(max / rate) * 1000) for x in real_delta]
    print(differences)
    # differences.insert(0, 0)

    # print("difference:", differences)

    # # guess_delta = []
    # # for i in range(9):
    # #     delta.append(mtest[i + 1] - mtest[i])
    # # mtest.sort()

    # print("Real delta:", real_delta)
    # print("Theoretical time per scan", max / rate)
    # print("Theoretical gap:", (sum(real_delta) / (int(len(timings)) - 1)))
    # print(len(timings))
    # print("\n")

    return differences

def plot_chemical_name(name, x_position, y_position, va, ha):
    plt.text(
        x_position,
        y_position,
        name,
        rotation_mode="default",
        verticalalignment=va,
        horizontalalignment=ha,
    )

def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = rga_scan_data_array[:, :, MASS_AMU_LAYER_INDEX]
    y = rga_scan_data_array[:, :, INTENSITY_TORR_LAYER_INDEX]
    z = rga_scan_data_array[:, :, TIME_SECONDS_LAYER_INDEX]

    surf = ax.plot_surface(x, z, y, cmap='summer', rstride=1, cstride=1, alpha=None, linewidth=0, antialiased=False)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")


pressure_delta_calc(time_stamps, scan_rate, stop_mass)

import_to_array()

f = open("demofile3.txt", "w")
f.write(str(rga_scan_data_array[:, 0, TIME_SECONDS_LAYER_INDEX]))
f.close()

print(len(rga_scan_data_array[:, 0, TIME_SECONDS_LAYER_INDEX]))
print(rga_scan_data_array[:, 0, TIME_SECONDS_LAYER_INDEX])
print(rga_scan_data_array[:, 1, TIME_SECONDS_LAYER_INDEX])
print(rga_scan_data_array[:, 2, TIME_SECONDS_LAYER_INDEX])
print(rga_scan_data_array[:, 3, TIME_SECONDS_LAYER_INDEX])
print(rga_scan_data_array[:, 4, TIME_SECONDS_LAYER_INDEX])

x = amu_layer_vector
y = rga_scan_data_array[:, 3, INTENSITY_TORR_LAYER_INDEX]

peaks_index, _ = sps.find_peaks(
    rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX],
    prominence=1 * (10 ** (-8)),
    distance=7,
    height=1 * (10 ** (-8)),
)  # , height=6* (10 ** (-8)))

peaks_index_thresholded, _ = sps.find_peaks(
    rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX],
    prominence=1 * (10 ** (-8)),
    distance=7,
    height=2.5 * (10 ** (-8)),
)

np.array(peaks_index)

plt.figure(figsize=(18, 6))
plt.plot(x, y)  # Plot some data on the Axes.
plt.scatter(x[peaks_index], y[peaks_index])
# plt.yscale("log")

plt.xlabel("Mass (Amu)")  # Add an x-label to the Axes.
plt.xticks(np.arange(start_mass - 1, stop_mass + 1, 5))
plt.xticks(np.arange(start_mass - 1, stop_mass + 1, 1), minor=True)

# plt.yticks(np.arange(0, max(t), 1), minor=True)

# ax.set_xticks(np.arange(start_mass,stop_mass,6),minor=True)
plt.ylabel("intensity (Torr)")  # Add a y-label to the Axes.
plt.title("Test Plot")  # Add a title to the Axes.
plt.grid()

# plt.text((x[peaks_index])[7], (y[peaks_index])[7], (x[peaks_index])[7])))

for i in range(len(x[peaks_index_thresholded])):
    plt.text(
        (x[peaks_index_thresholded])[i],
        (y[peaks_index_thresholded])[i] + (3 * (10 ** (-8))),
        round(((x[peaks_index_thresholded])[i]), 1),
        rotation=90,
        rotation_mode="anchor",
        verticalalignment="center",
        horizontalalignment="left",
    )

plot_chemical_name("Oxygen", 32, (1.56 * (10 ** (-7))), va="center", ha="left")

# plt.savefig("graph", dpi=250, bbox_inches="tight")

plot_3d()

plt.show()
