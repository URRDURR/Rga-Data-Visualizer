import os
import sys
import numpy as np
import pandas as pan
import matplotlib.pyplot as plt
import scienceplots
import data_file_reader
# import plotly.express as px
(plt.style.use("science"))

# np.set_printoptions(threshold=1000000000000, linewidth=1000000000000)


folder_path = r"D:\-SFU Physics Lab Work\HIM Room\Guneet\2024-07-11\RGA\Scan 1 - Testing Fillament Pressure Increase - 2024_07_11\Analog-20240711-114931-552.rgadata"

time_stamps, spectra, total_pressures, json_file, number_of_cycles = data_file_reader.read_file_data(folder_path)

json_split = json_file.replace(",", "").split("\n")

points_per_amu = int((json_split[30])[28:])
scan_rate = int((json_split[31])[24:])
stop_mass = int((json_split[32])[25:])
start_mass = int((json_split[33])[24:])

# print("points_per_amu", points_per_amu)
# print("scan_rate", scan_rate)
# print("stop_mass", stop_mass)
# print("start_mass", start_mass)


def import_to_array():

    global rga_scan_data_array

    global amu_layer_vector

    global MASS_AMU_LAYER_INDEX
    MASS_AMU_LAYER_INDEX = 1

    global INTENSITY_TORR_LAYER_INDEX
    INTENSITY_TORR_LAYER_INDEX = 2

    global TIME_SECONDS_LAYER_INDEX
    TIME_SECONDS_LAYER_INDEX = 3

    # The array has a height of 3 to accomadate the AMU value, the time, and the pressure
    LAYER_DIMENSIONS = 3

    number_of_amu_points = ((start_mass - stop_mass) * points_per_amu) + 1

    # array initialized for the right amount of space
    rga_scan_data_array = np.ones((number_of_amu_points, number_of_cycles, LAYER_DIMENSIONS))

    # its maybe more convinent to have the AMU position bundled in with the main array, may change if any preformance/ease of use issues occour
    amu_layer_vector = np.linspace(start_mass, stop_mass, number_of_amu_points)
    amu_layer_array = amu_layer_vector[:, np.newaxis]
    amu_layer_array = np.repeat(amu_layer_array, number_of_cycles, 1)
    rga_scan_data_array[:, :, MASS_AMU_LAYER_INDEX] = amu_layer_array
    
    for i in range(number_of_cycles):
        rga_scan_data_array[:, i, INTENSITY_TORR_LAYER_INDEX] = spectra[i]

    rga_scan_data_array[:, :, MASS_AMU_LAYER_INDEX] = amu_layer_array



def pressure_delta_calc(timings, rate, max):

    print("Real timings:", timings)

    # guess = []
    # for i in range(len(timings)-1):
    #     guess.append(timings[] + i * 208)

    real_delta = []
    for i in range(len(timings) - 1):
        real_delta.append(timings[i + 1] - timings[i])

    # guess_delta = []
    # for i in range(9):
    #     delta.append(mtest[i + 1] - mtest[i])
    # mtest.sort()

    print("Real delta:", real_delta)
    print("Theoretical time per scan", max / rate)
    print("Theoretical gap:", (timings[1] - timings[0]))

    print("\n")


import_to_array()


x = amu_layer_vector
y = rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX]

print(sum(y) * 0.0028)

plt.figure(1)
plt.plot(x, y)

x = np.linspace(1, 10, 10)
# print(x)
y = [7, 215, 423, 630, 838, 1046, 1254, 1461, 1669, 1877]


hi = [2, 17, 32, 46, 61, 76, 91, 106, 120, 135, 150, 165, 179, 194, 209, 224, 239, 253, 268, 283]


# plt.figure(2)
# plt.plot(x, y)


plt.show()
