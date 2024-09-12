# The code excerpt below is a sample to read a RGA data file in Python.
# The sample code is based on a data file that has 2 step scan sequence:
#   Step 1: An Analog or a Histogram scan.
#   Step 2: PvsT scan with various gases.
# The code will extract all data of the Analog/Histogram scan and optionally
# skip over the PvsT scan (change flag SKIP_STEP2_DATA)
#
# For an Analog or a Histogram scan:
#   number_of_data_points = (stop_mass - start_mass) * points_per_amu + 1
#   points_per_amu = 1 for a Histogram scan
#
# For a PvsT scan:
#   number_of_data_points = number_of_gases
#
# With an installed Python 3.6+, run the script as follows:
# python <directory>\dataFileReader.py "<data_directory>\<file_name>.rgadata"

import sys
import os
import struct
import json

# Added this to make it easier to use bits of the code without having to change every print statment
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def read_int(fd):
    bytes = fd.read(4)
    value = struct.unpack('i', bytes)[0]
    return value

def read_uint(fd):
    bytes = fd.read(4)
    value = struct.unpack('I', bytes)[0]
    return value

def read_int64(fd):
    bytes = fd.read(8)
    value = struct.unpack('q', bytes)[0]
    return value
    
def read_float(fd):
    bytes = fd.read(4)
    value = struct.unpack('f', bytes)[0]
    return value
    
def read_boolean(fd):
    bytes = fd.read(1)
    value = struct.unpack('?', bytes)[0]
    return value


SKIP_STEP2_DATA = False

def read_file_data(path):

    print("see???")

    # with HiddenPrints(): # don't have to manualy disable prints now, can just get needed variables

    with open(path, "rb") as f:
        # File description - 32-byte string
        f_identifier = f.read(32)
        print('File identifier:', f_identifier)
        
        # File version - int32 value
        f_version = read_int(f)
        print('File version:', f_version)
        
        # Single Precision Floating Point - boolean
        is_single_precision = read_boolean(f)
        print('Is single precision floating point:', is_single_precision)
        
        # Metadata - QVector of int64 - vector size = 100
        vsize = read_uint(f)
        print('Metadata vsize =', vsize)
        bytes = f.read(vsize * 8)  # vector values
        metadata_list = [struct.unpack('q', bytes[8*i:8*i+8])[0] for i in range(vsize)]
        print('Metadata =', metadata_list)
        
        # Decode Metadata
        settings_location = metadata_list[0]
        data_location = metadata_list[1]
        settings_size = metadata_list[2]  # size of (JSON) settings block
        data_size = metadata_list[3]  # size of data block
        number_of_cycles = metadata_list[4]  # how many cycles of scan data
        single_cycle_data_size = metadata_list[5]  # size of a single cycle
        number_of_active_scan_steps = metadata_list[6]
        upper_index = 7 + number_of_active_scan_steps
        step_data_sizes = metadata_list[7:upper_index]  # data size of individual steps
            # Calculation:
            # data_size = single_cycle_data_size * number_of_cycles
            # single_cycle_data_size = SumOf(step_data_sizes) + Auxiliary_signal_sizes
        
        # The Settings in the JSON format can give the details of the scan sequence:
        #  How many steps, what kind of scan in each step, what settings for
        #  the mass range (Start/Stop mass), or how many individual mass used
        #  in the PvsT mode, etc.

        # Skip to the settings
        f.seek(settings_location)
        bytes = f.read(settings_size)
        json_string_len = struct.unpack('i', bytes[0:4])[0]
        print('JSON string length =', json_string_len)
        bytes = bytes[4:]  # slice to get the content portion
        json_string = bytes.decode('utf-8')
        json_settings = json.loads(json_string)
        print(json.dumps(json_settings, indent=4))
        json_file = (json.dumps(json_settings, indent=4))
        
        # Decode auxiliary signal query options
        # Aux signals are not available in version 17 or smaller
        # (associated with released version 4.0.x-4.1.x)
        if f_version > 17:
            ss_query_options = json_settings['ssQueryOpts']
            print('\nAuxiliary signal query options =', ss_query_options)
            # Option value: 0 - Disabled (no reading), 1 - Enabled (read once per cycle)
            print('Query Option - Total pressure:', ss_query_options['1000'])
            print('Query Option - RTD temperature:', ss_query_options['1100'])
            print('Query Option - Flange temperature:', ss_query_options['1200'])
            print('Query Option - Analog Vin:', ss_query_options['1300'])
            print('Query Option - Analog Iin:', ss_query_options['1400'])
            print('Query Option - GPIO in:', ss_query_options['1500'])
        
        # Skip to the scan data
        f.seek(data_location)

        time_stamps = []
        spectra = []
        pvst = []
        total_pressures = []
        rtd_temperatures = []
        flange_temperatures = []
        analog_Vin_signals = []
        analog_Iin_signals = []
        gpio_in_signals = []
        for cycle in range(number_of_cycles):
            print('\ncycle =', cycle + 1)
            
            # Auxiliary signals
            if f_version > 17:
                # Total pressure
                total_pressure = read_float(f)
                total_pressures.append(total_pressure)
                print('Total pressure =', total_pressure)
                
                # RTD temperature
                rtd_t = read_float(f)
                rtd_temperatures.append(rtd_t)
                print('RTD temperature =', rtd_t)
                
                # Flange temperature
                flange_t = read_float(f)
                flange_temperatures.append(flange_t)
                print('Flange temperature =', flange_t)
                
                # Analog Vin
                analog_Vin = read_float(f)
                analog_Vin_signals.append(analog_Vin)
                print('Analog Vin =', analog_Vin)
                
                # Analog Iin
                analog_Iin = read_float(f)
                analog_Iin_signals.append(analog_Iin)
                print('Analog Iin =', analog_Iin)
                
                # GPIO input
                gpio_in = read_int(f)
                gpio_in_signals.append(gpio_in)
                print('GPIO input =', gpio_in)
                print()
            
            # Step data
            for step in range(len(step_data_sizes)):
                print('Step ', step + 1)
                # Time stamp
                time_stamp = read_int64(f)  # in ms
                print('Time stamp =', time_stamp, 'ms')
                time_stamps.append(time_stamp)
                
                if (step == 0):  # Analog/Histogram scan step
                    # Signal intensities - QVector of float
                    vsize = read_uint(f)  # vsize = ((stop_mass - start_mass) / points_per_amu) + 1
                    print('vsize (number of data points) =', vsize)
                    bytes = f.read(vsize * 4)
                    scan_signals = []
                    scan_signals = [struct.unpack('f', bytes[4*i:4*i+4])[0] for i in range(vsize)]
                    spectra.append(scan_signals)
                    print('Spectrum signals =', scan_signals)
                elif (step == 1):  # PvsT scan step
                    # Extract the number of gases from JSON
                    json_step2 = json_settings['cfgs'][1]
                    json_gases = json_step2['gases']
                    n_gases = len(json_gases)
                    print('PvsT number of gases =', n_gases)
                    print('PvsT gases =', json_gases)
                    # Signal intensities - N gases: N float values
                    scan_signals = []
                    bytes = f.read(n_gases * 4)
                    scan_signals = [struct.unpack('f', bytes[4*i:4*i+4])[0] for i in range(n_gases)]
                    pvst.append(scan_signals)
                    print('PvsT signals =', scan_signals)
            
                if SKIP_STEP2_DATA:
                    # Skip next step's data
                    second_step_data_size = step_data_sizes[1]
                    f.seek(second_step_data_size, 1)  # skip N bytes from the current file position
                    break

                        
        print('Time stamps =', time_stamps)
        print('Analog/Histogram signal itensities =', spectra)
        print('PvsT signal itensities =', pvst)
        print('Total pressures =', total_pressures)
        print('RTD temperatures =', rtd_temperatures)
        print('Flange temperatures =', flange_temperatures)
        print('Analog Vin signals =', analog_Vin_signals)
        print('Analog Iin signals =', analog_Iin_signals)
        print('GPIO input signals =', gpio_in_signals)
 
        return(time_stamps,spectra,total_pressures,json_file,number_of_cycles)
