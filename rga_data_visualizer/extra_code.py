import os

# give the locations of all files of a type in a folder
def extract_file_locations(folder_path,file_type):

    # variable holds list of locations of all files of one type in a folder
    global file_paths
    file_paths = []

    # creates list of all .{} files in directory
    testing = os.listdir(folder_path)
    # print(testing)

    for i in os.listdir(folder_path):
        if i.endswith(file_type) and i.startswith(r"Analog-"):
            file_paths.append(i)

    # files sorted to be chronological
    file_paths.sort()

    # give absolute path of each file in the directory
    for i in range(len(file_paths)):
        file_paths[i] = folder_path + "\\" + file_paths[i]

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

    number_of_amu_points = 

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