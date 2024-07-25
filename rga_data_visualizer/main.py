import os
import sys
import numpy as np
import pandas as pan
import matplotlib.pyplot as plt
import scipy.signal as sps

import scienceplots
import data_file_reader

import plotly.express as px

# (plt.style.use("science"))

# np.set_printoptions(threshold=1000000000000, linewidth=1000000000000)


folder_path = r"D:\-SFU Physics Lab Work\HIM Room\Guneet\2024-07-11\RGA\Scan 3 - Increased Points - 2024_07_11\Analog-20240711-123603-709.rgadata"

time_stamps, spectra, total_pressures, json_file, number_of_cycles = data_file_reader.read_file_data(folder_path)

json_split = json_file.replace(",", "").split("\n")
print(json_file)
points_per_amu = int((json_split[30])[28:])
scan_rate = float((json_split[31])[24:])
start_mass = int((json_split[32])[25:])
stop_mass = int((json_split[33])[24:])

print(start_mass)
print(stop_mass)
print(np.arange(0, 66, 5))
print(start_mass + 4 + stop_mass)

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

    number_of_amu_points = ((stop_mass - start_mass) * points_per_amu) + 1

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
    print("Theoretical gap:", (sum(real_delta) / (int(len(timings)) - 1)))
    print(len(timings))
    print("\n")


print(time_stamps)
my_new_list = [i * 0.001 for i in time_stamps]
pressure_delta_calc(my_new_list, scan_rate, stop_mass)

print(my_new_list)

import_to_array()


x = amu_layer_vector
y = rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX]

# noise_removal = sps.savgol_filter((rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX]),window_length= 6, polyorder=1)
# print(noise_removal)


def fft_noise_removal(noisy_signal):
    threshold = 0.0001 
    fhat = np.fft.fft(
        rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX],
        len(rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX]),
    )
    PSD = fhat * np.conj(fhat) / len(rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX])

    plt.figure()
    plt.plot(x, fhat)
    plt.title("1")

    plt.figure()
    plt.plot(x, PSD)
    plt.title("2")

    indices = PSD < (0.09*(10**(-10)))
    print(indices)
    PSDclean = PSD * indices
    fhat = indices * fhat

    plt.figure()
    plt.plot(x, fhat)
    plt.title("3")

    ffilt = np.fft.ifft(fhat)
    print("tesgggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggget",ffilt)
    return ffilt

#value = fft_noise_removal(y)

# plt.figure()
# plt.plot(x, value)
# plt.title("4")
print(len(y))

w = sps.savgol_filter(y, 7, 3)
plt.figure()
plt.plot(x, w)
plt.title("4")

z, _ = sps.find_peaks(rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX], prominence = 1 * (10 ** (-8)), distance = 7, height = 1 * (10 ** (-8)))#, height=6* (10 ** (-8)))

np.array(z)
# 0.0001
# plt.figure()
# plt.plot(x, fft_tran)
# plt.yscale("log")

print(sum(y) * 0.0028)
print(np.shape(z))
# fig, ax = plt.subplots()  # Create a figure containing a single Axes.
print(type(z))
print(rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX])
t = rga_scan_data_array[:, 1, INTENSITY_TORR_LAYER_INDEX]
t = t[np.array(z)]
print(t)


plt.figure()
plt.plot(x, y)  # Plot some data on the Axes.
plt.scatter(x[z], y[z])
# plt.yscale("log")

plt.xlabel("Mass (Amu)")  # Add an x-label to the Axes.
plt.xticks(np.arange(start_mass - 1, stop_mass + 1, 5))
plt.xticks(np.arange(start_mass - 1, stop_mass + 1, 1), minor=True)

#plt.yticks(np.arange(0, max(t), 1), minor=True)

# ax.set_xticks(np.arange(start_mass,stop_mass,6),minor=True)
plt.ylabel("intensity (Torr)")  # Add a y-label to the Axes.
plt.title("Test Plot")  # Add a title to the Axes.
plt.grid()

#plt.tick_params(direction="out")

for i in range(len(x[z])):
    print((x[z])[i], (y[z])[i])

# plt.figure()
# plt.plot(x, y)
# plt.xlabel("Mass (Amu)")
# plt.ylabel("intensity (Torr)")
# plt.xticks(start_mass, stop_mass, 5)
# plt.yscale("log")
# plt.tick_params()

x = np.linspace(1, 10, 10)
# print(x)
y = [7, 215, 423, 630, 838, 1046, 1254, 1461, 1669, 1877]


hi = [2, 17, 32, 46, 61, 76, 91, 106, 120, 135, 150, 165, 179, 194, 209, 224, 239, 253, 268, 283]


# plt.figure(2)
# plt.plot(x, y)


plt.show()
