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

print(time_stamps)
# print(spectra)
print(total_pressures)
# print(json_file)

json_split = json_file.replace(",", "").split("\n")

# print(json_split, 2*"\n")

points_per_amu = int((json_split[30])[28:])
scan_rate = int((json_split[31])[24:])
stop_mass = int((json_split[32])[25:])
start_mass = int((json_split[33])[24:])

print("points_per_amu", points_per_amu)
print("scan_rate", scan_rate)
print("stop_mass", stop_mass)
print("start_mass", start_mass)


def import_to_array():

    global main_array

    global MASS_AMU
    MASS_AMU = 1

    global INTENSITY_TORR
    INTENSITY_TORR = 2

    global TIME_SECONDS
    TIME_SECONDS = 3

    # extracted rga parameters for setting up arrays and giving additional data to plots
    # scan_parameters = open(file_paths[0], "r")
    # scan_parameters = scan_parameters.read().splitlines()[:25]
    # print(scan_parameters)

    # number_of_amu_points = int((scan_parameters[6])[23:])
    # number_of_files = len(file_paths)

    # The array has a height of 3 to accomadate the AMU value, the time, and the pressure
    DIMENSIONS = 3

    number_of_amu_points = ((start_mass - stop_mass) * points_per_amu) + 1
    print(number_of_amu_points)

    main_array = np.ones((number_of_amu_points, number_of_cycles, DIMENSIONS))
    # print(main_array)
    main_array[:, :, 1]

    thing_array = np.linspace(
        1,
        100,
        number_of_amu_points,
    )

    thing_array = thing_array[:, np.newaxis]
    # print(thing_array)
    thing_array = np.repeat(thing_array, number_of_cycles, 1)

    # print(np.shape(main_array))
    # print(np.shape(thing_array))
    # print(thing_array)
    # print((main_array[:,:,1]))
    # main_array[:, :, 1] = thing_array
    # print(np.shape(main_array[:, :, 1]))
    # np.savetxt("matrix_output.txt", thing_array)
    # print(main_array[:, :, 1])
    # print(np.shape(main_array[:, :, 1]))

    for i in range(len(file_paths)):

        main_data = open(file_paths[i], "r")
        main_data = main_data.read().replace(" ", "").replace(",", "\n").splitlines()[57:]
        # print(main_data[1::2])
        # print(main_array[:,i,2])
        main_array[:, i, 2] = main_data[1::2]
        # print(s)
        # print(len(s))

        # print("\n" * 3)


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


x = main_array[:, 1, MASS_AMU]
y = main_array[:, 1, INTENSITY_TORR]

print(sum(y) * 0.0028)

plt.figure(1)
plt.plot(x, y)

x = np.linspace(1, 10, 10)
# print(x)
y = [7, 215, 423, 630, 838, 1046, 1254, 1461, 1669, 1877]
# guess = []
# for i in range(10):
#     guess.append(7 + i * 208)
# real_delta = []
# for i in range(9):
#     real_delta.append(y[i + 1] - y[i])
# # mtest.sort()
# print(guess)
# print(real_delta)
# print(100 / 0.5)
# print("\n")

pressure_delta_calc(y, 0.5, 100)

hi = [2, 17, 32, 46, 61, 76, 91, 106, 120, 135, 150, 165, 179, 194, 209, 224, 239, 253, 268, 283]

# print(hi)

# hi2 = []
# for i in range(19):
#     hi2.append(hi[i + 1] - hi[i])
# print(hi2)
# print(65 / 5)

pressure_delta_calc(hi, 5, 65)

plt.figure(2)
plt.plot(x, y)


# plt.show()

"""
file = open('D:/Adobe/text.txt', "r+")
for line in file.readlines():
    print(line)
file.close()
"""
