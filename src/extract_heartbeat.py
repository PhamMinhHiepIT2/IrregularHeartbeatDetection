# File to save images of beats in a specified directory

import os
import wfdb
import argparse
import signal_api
import directory_structure
import natsort  # module used to sort file names
from concurrent.futures import ProcessPoolExecutor


def process_signal(signal_path):
    # get annotation data frame of signal file
    ann = wfdb.rdann(signal_path, 'atr', return_label_elements=[
        'symbol', 'description', 'label_store'], summarize_labels=True)
    # uncomment to save images of beats
    signal_api.extractBeatsFromPatient(signal_path, ann)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--cpu', type=int, default=12,
                      help='number of cpu to use')

    args = args.parse_args()
    cpu = args.cpu
    print(cpu)

    DATA_DIR = 'data/mit-bih_waveform/'

    # get all .hea and .dat files (respectively)
    signal_files = directory_structure.filesInDirectory('.hea', DATA_DIR)

    print(signal_files)

    # sort file names in ascending order in list
    signal_files = natsort.natsorted(signal_files)

    # extract and save beats from file provided
    with ProcessPoolExecutor(cpu) as executor:
        for signal_file in signal_files:
            signal_path = DATA_DIR + \
                directory_structure.removeFileExtension(signal_file)
            print(signal_path)

            # get annotation data frame of signal file
            executor.submit(process_signal, signal_path)
