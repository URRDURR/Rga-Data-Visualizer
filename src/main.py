import json
from operator import itemgetter
from tkinter import filedialog as fd
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sps
from scipy.optimize import curve_fit

import data_file_reader

# import scienceplots
# import plotly.express as px

# (plt.style.use("science"))

# The array has a height of 3 to accomadate the AMU value, the time, and the pressure
MASS_AMU_INDEX = 0
INTENSITY_TORR_INDEX = 1
TIME_SECONDS_INDEX = 2
LAYER_DIMENSIONS = 3


def import_to_array(start_mass, stop_mass, points_per_amu, number_of_cycles):
    number_of_amu_points = ((stop_mass - start_mass) * points_per_amu) + 1

    # array initialized for the right amount of space
    rga_scan_data_array = np.ones((number_of_amu_points, number_of_cycles, LAYER_DIMENSIONS))

    # its maybe more convinent to have the AMU position bundled in with the
    # main array, may change if any preformance/ease of use issues occour
    amu_layer_vector = np.linspace(start_mass, stop_mass, number_of_amu_points)
    amu_layer_array = amu_layer_vector[:, np.newaxis]
    amu_layer_array = np.repeat(amu_layer_array, number_of_cycles, 1)
    rga_scan_data_array[:, :, MASS_AMU_INDEX] = amu_layer_array

    difference = pressure_delta_calc(time_stamps, scan_rate, stop_mass)

    for i in range(number_of_cycles):
        rga_scan_data_array[:, i, INTENSITY_TORR_INDEX] = spectra[i]
        for n in range(number_of_amu_points):
            rga_scan_data_array[n, i, TIME_SECONDS_INDEX] = n * 200_000 / number_of_amu_points + sum(difference[:i]) + 200_000 * i

    return rga_scan_data_array


def pressure_delta_calc(timings, rate, maximum):
    # timings = [x/1000 for x in timings]
    # print("Real timings:", timings)

    # guess = []
    # for i in range(len(timings)-1):
    #     guess.append(timings[] + i * 208)

    real_delta = []
    for i in range(len(timings) - 1):
        real_delta.append(timings[i + 1] - timings[i])

    differences = [x - (int(maximum / rate) * 1000) for x in real_delta]
    # print(differences)
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


def plot_3d(scan_data,title):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)

    x = scan_data[:, :, MASS_AMU_INDEX]
    y = scan_data[:, :, INTENSITY_TORR_INDEX]
    z = scan_data[:, :, TIME_SECONDS_INDEX]

    surf = ax.plot_surface(x, z, y, cmap="summer", rstride=1, cstride=1, alpha=None, linewidth=0, antialiased=False)

    ax.set_xlabel("x: Mass (AMU)")
    ax.set_ylabel("z: Time (Milliseconds)")
    ax.set_zlabel("y: Pressure (Torr)")


def plot_2d(scan_data, scan_number, title):
    x = scan_data[:, scan_number, MASS_AMU_INDEX]
    y = scan_data[:, scan_number, INTENSITY_TORR_INDEX]

    peaks, _ = sps.find_peaks(
        scan_data[:, scan_number, INTENSITY_TORR_INDEX],
        prominence=1 * (10 ** (-8)),
        distance=7,
        height=1 * (10 ** (-8)),
    )

    # primary_peaks = []
    # for i in peaks:
    #     if scan_data[i, scan_number, INTENSITY_TORR_INDEX] >= 2.5 * (10 ** (-8)):
    #         primary_peaks.append(i)
    # primary_peaks = np.array(primary_peaks)

    PEAK_SIGNIFICANCE_THRESHOLD = 2.5 * (10 ** (-8))

    significant_peaks = np.array(
        [(peak_index) for peak_index in peaks if (scan_data[peak_index, scan_number, INTENSITY_TORR_INDEX] >= PEAK_SIGNIFICANCE_THRESHOLD)]
    )

    plt.figure(figsize=(18, 6))
    plt.plot(x, y)  # Plot some data on the Axes.
    plt.scatter(x[peaks], y[peaks])
    # plt.yscale("log")
    plt.xlabel("Mass (Amu)")  # Add an x-label to the Axes.
    plt.xticks(np.arange(math.floor(start_mass / 5) * 5, math.ceil(stop_mass / 5) * 5, 5))
    plt.xticks(np.arange(math.floor(start_mass / 5) * 5, math.ceil(stop_mass / 5) * 5, 1), minor=True)
    # plt.yticks(np.arange(0, max(t), 1), minor=True)
    # ax.set_xticks(np.arange(start_mass,stop_mass,6),minor=True)
    plt.ylabel("intensity (Torr)")  # Add a y-label to the Axes.
    plt.title(title)  # Add a title to the Axes.
    plt.grid()
    plt.ylim(top=max(y[significant_peaks]) * 1.25)

    print(max(y[significant_peaks]) * 1.15)
    print(stop_mass)
    print(math.ceil(stop_mass / 5) * 5)

    # plt.text((x[peaks_index])[7], (y[peaks_index])[7], (x[peaks_index])[7])
    # for i in len(molocule_json_output["molecules"]):
    #     m = molocule_json_output["molecules"][i]["mass"]

    # values = set(m) & set(peaks_index_thresholded)

    significant_peak_masses = x[significant_peaks]
    significant_peak_pressures = y[significant_peaks]

    for i, significant_peak_mass in enumerate(significant_peak_masses):
        name = ""

        for _, compound in enumerate(compounds_dictionary):
            if significant_peak_mass - 0.5 <= compound["mass"] <= significant_peak_mass + 0.5:
                print(significant_peak_mass)
                if name != "":
                    name += "/" + compound["name"]
                else:
                    name = " " + compound["name"]
        print(name)

        plt.text(
            significant_peak_mass,
            significant_peak_pressures[i] + (3 * (10 ** (-8))),
            str(round((significant_peak_mass), 1)) + name,
            rotation=90,
            rotation_mode="anchor",
            verticalalignment="center",
            horizontalalignment="left",
            weight="bold",
        )

        # plt.savefig("graph", dpi=250, bbox_inches="tight")


# TODO:
# 1. make it work for total pressure as well
# 2. add selection options
# 3. test agains other samples
def halflife(scan_data, time_stamps, total_pressures, title):

    value = scan_data[10, :, :]
    print(value[1:2, :])

    def funct(x1, a, b):
        return a * x1 + b

    xdata = np.linspace(min(value[:, 2]), max(value[:, 2]), num=17000)

    xdata2 = np.linspace(min(time_stamps), max(time_stamps), num=17000)

    ydata, other = curve_fit(funct, np.squeeze(value[:, 2]), np.squeeze(value[:, 1]), maxfev=1000000)
    test = sum(scan_data[:, :, :])
    print(np.shape(value))
    dfo = pd.DataFrame(value)
    print(dfo)

    print(time_stamps)
    print(total_pressures)
    print(type(time_stamps))
    print(type(total_pressures))

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    advancd_curve, other_2 = curve_fit(funct, np.array(time_stamps), np.array(total_pressures), maxfev=1000000)
    print("adv. curve", advancd_curve)

    print(test, "\n")
    print(ydata, "\n")
    print(other, "\n")

    plt.scatter(np.squeeze(value[:, 2]), np.squeeze(value[:, 1]))
    plt.plot(xdata, funct(xdata, *ydata))
    plt.title("Scan 1 - Testing Fillament Pressure Increase - 2024_07_11 - H2 Decay")
    plt.ylabel("intensity (Torr)")
    plt.xlabel("Time (Miliseconds)")
    plt.show()

    plt.scatter(time_stamps, total_pressures)
    plt.plot(xdata2, funct(xdata2, *advancd_curve))
    plt.title("Scan 1 - Testing Fillament Pressure Increase - 2024_07_11 - Chamber Pressure Decay ")
    plt.ylabel("intensity (Torr)")
    plt.xlabel("Time (Miliseconds)")
    plt.show()

    dfo.to_csv("file4.csv", index=False)


# Currently being done under the assumption that we are only using the 3rd scan
PATH_VIA_TERMINAL = False

if PATH_VIA_TERMINAL:
    print("Enter folder location")
    file_path = fd.askopenfilename()
else:
    file_path = r"C:\Users\Big Me\OneDrive - Simon Fraser University (1sfu)\KavanaghLab\02-Personal_Folders\Guneet Malhotra\HIM Room\Guneet\2024-06-27\RGA\Scan 3 - Slow Scan\Analog-20240627-141459-709.rgadata"

time_stamps, spectra, total_pressures, json_file, number_of_cycles = data_file_reader.read_file_data(file_path)

print(time_stamps)
print(total_pressures)

rga_json_data = json.loads(json_file)
# points_per_amu = rga_json_data["cfgs"][0]["pointsPerAmu"]
# scan_rate = rga_json_data["cfgs"][0]["scanRate"]
# start_mass = rga_json_data["cfgs"][0]["startMass"]
# stop_mass = rga_json_data["cfgs"][0]["stopMass"]

points_per_amu, scan_rate, start_mass, stop_mass = itemgetter("pointsPerAmu", "scanRate", "startMass", "stopMass")(
    rga_json_data["cfgs"][0]
)

print(os.listdir())

with open("molocule.json", encoding="utf-8") as file:
    compounds_dictionary = json.load(file)

rga_scan_data = import_to_array(start_mass, stop_mass, points_per_amu, number_of_cycles)

while True:
    selection = input("\nwould you like a 2D plot [1], 3D plot [2], pressure decay [3], or exit [4]?: ")
    if selection == "1":
        while True:
            subscan = input("which sub-scan would you like to graph?: ")
            if subscan.isnumeric() and (int(subscan) > number_of_cycles):
                print("number is larger than number of scans (", number_of_cycles, "), please try again\n")
                continue
            elif subscan.isnumeric() and (int(subscan) <= number_of_cycles):
                subscan = int(subscan) - 1
                break
            else:
                print("please try again\n")
        title = input("Title?: ")
        plot_2d(rga_scan_data, subscan, title)

        plt.show()
        continue

    elif selection == "2":
        title = input("Title?: ")
        plot_3d(rga_scan_data, title)
        plt.show()

    elif selection == "3":
        title = input("Title?: ")
        halflife(rga_scan_data, time_stamps, total_pressures, title)
    elif selection == "4":
        break
    else:
        print("please try again")
