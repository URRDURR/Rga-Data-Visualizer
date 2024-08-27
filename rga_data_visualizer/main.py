import os
import sys
import numpy as np
import pandas as pan
import matplotlib.pyplot as plt
import scipy.signal as sps
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import json
from tkinter import filedialog as fd
import data_file_reader
import pandas as pd

# import scienceplots
# import plotly.express as px

# (plt.style.use("science"))


def import_to_array(start_mass, stop_mass, points_per_amu, number_of_cycles):

    number_of_amu_points = ((stop_mass - start_mass) * points_per_amu) + 1

    # array initialized for the right amount of space
    rga_scan_data_array = np.ones((number_of_amu_points, number_of_cycles, LAYER_DIMENSIONS))

    # its maybe more convinent to have the AMU position bundled in with the main array, may change if any preformance/ease of use issues occour
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


def pressure_delta_calc(timings, rate, max):

    # timings = [x/1000 for x in timings]
    # print("Real timings:", timings)

    # guess = []
    # for i in range(len(timings)-1):
    #     guess.append(timings[] + i * 208)

    real_delta = []
    for i in range(len(timings) - 1):
        real_delta.append(timings[i + 1] - timings[i])

    differences = [x - (int(max / rate) * 1000) for x in real_delta]
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


def plot_3d(scan_data):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = scan_data[:, :, MASS_AMU_INDEX]
    y = scan_data[:, :, INTENSITY_TORR_INDEX]
    z = scan_data[:, :, TIME_SECONDS_INDEX]

    surf = ax.plot_surface(x, z, y, cmap="summer", rstride=1, cstride=1, alpha=None, linewidth=0, antialiased=False)

    ax.set_xlabel("x: Mass (AMU)")
    ax.set_ylabel("z: Time (Seconds)")
    ax.set_zlabel("y: Pressure (Torr)")


def plot_2d(scan_data, scan_number):
    x = scan_data[:, scan_number, MASS_AMU_INDEX]
    y = scan_data[:, scan_number, INTENSITY_TORR_INDEX]
    print(scan_number)
    peaks, _ = sps.find_peaks(
        scan_data[:, scan_number, INTENSITY_TORR_INDEX],
        prominence=1 * (10 ** (-8)),
        distance=7,
        height=1 * (10 ** (-8)),
    )

    primary_peaks = []
    for i in peaks:
        if scan_data[i, scan_number, INTENSITY_TORR_INDEX] >= 2.5 * (10 ** (-8)):
            primary_peaks.append(i)
    primary_peaks = np.array(primary_peaks)

    plt.figure(figsize=(18, 6))
    plt.plot(x, y)  # Plot some data on the Axes.
    plt.scatter(x[peaks], y[peaks])
    # plt.yscale("log")
    plt.xlabel("Mass (Amu)")  # Add an x-label to the Axes.
    plt.xticks(np.arange(start_mass - 1, stop_mass + 1, 5))
    plt.xticks(np.arange(start_mass - 1, stop_mass + 1, 1), minor=True)
    # plt.yticks(np.arange(0, max(t), 1), minor=True)
    # ax.set_xticks(np.arange(start_mass,stop_mass,6),minor=True)
    plt.ylabel("intensity (Torr)")  # Add a y-label to the Axes.
    plt.title("Test Plot")  # Add a title to the Axes.
    plt.grid()

    # plt.text((x[peaks_index])[7], (y[peaks_index])[7], (x[peaks_index])[7])
    # for i in len(molocule_json_output["molecules"]):
    #     m = molocule_json_output["molecules"][i]["mass"]

    # values = set(m) & set(peaks_index_thresholded)

    for i in range(len(x[primary_peaks])):
        name = ""
        for n in range(len(molocule_json_output["molecules"])):
            if (x[primary_peaks])[i] + 0.5 >= molocule_json_output["molecules"][n]["mass"] and (x[primary_peaks])[
                i
            ] - 0.5 <= molocule_json_output["molecules"][n]["mass"]:
                if name != "":
                    name += "/" + molocule_json_output["molecules"][n]["name"]
                    continue
                name = " " + molocule_json_output["molecules"][n]["name"]

        plt.text(
            (x[primary_peaks])[i],
            (y[primary_peaks])[i] + (3 * (10 ** (-8))),
            str(round(((x[primary_peaks])[i]), 1)) + name,
            rotation=90,
            rotation_mode="anchor",
            verticalalignment="center",
            horizontalalignment="left",
        )

        # plt.savefig("graph", dpi=250, bbox_inches="tight")


def halflife():
    pass


# Currently being done under the assumption that we are only using the 3rd scan
# The array has a height of 3 to accomadate the AMU value, the time, and the pressure
MASS_AMU_INDEX = 0
INTENSITY_TORR_INDEX = 1
TIME_SECONDS_INDEX = 2
LAYER_DIMENSIONS = 3
PATH_VIA_TERMINAL = False

if PATH_VIA_TERMINAL == True:
    print("Enter folder location")
    file_path = fd.askopenfilename()
else:
    file_path = r"C:\Users\gma78\OneDrive - Simon Fraser University (1sfu)\KavanaghLab\02-Personal_Folders\Guneet Malhotra\HIM Room\Guneet\2024-07-11\RGA\Scan 3 - Increased Points - 2024_07_11\Analog-20240711-123603-709.rgadata"

time_stamps, spectra, total_pressures, json_file, number_of_cycles = data_file_reader.read_file_data(file_path)

rga_json_data = json.loads(json_file)
points_per_amu = rga_json_data["cfgs"][0]["pointsPerAmu"]
scan_rate = rga_json_data["cfgs"][0]["scanRate"]
start_mass = rga_json_data["cfgs"][0]["startMass"]
stop_mass = rga_json_data["cfgs"][0]["stopMass"]

f = open("molocule.json")
molocule_json_output = json.load(f)
f.close()

rga_scan_data = import_to_array(start_mass, stop_mass, points_per_amu, number_of_cycles)

# This is bad code I do not know how to fix
while True:
    selection = input("\nwould you like a 2D plot [1], 3D plot [2], halflife data [3], or exit [4]?: ")
    if selection == "1":
        while True:
            subscan = input("which sub-scan would you like to graph?: ")
            if subscan.isnumeric():
                if int(subscan) <= number_of_cycles:
                    subscan = int(subscan) - 1
                    break
                print("number is larger than number of scans, please try again\n")
                continue
            print("please try again\n")
        plot_2d(rga_scan_data, subscan)
        plt.show()
        continue

    elif selection == "2":
        plot_3d(rga_scan_data)
        plt.show()

    elif selection == "3":
        print("todo")
    elif selection == "4":
        break
    else:
        print("please try again")
