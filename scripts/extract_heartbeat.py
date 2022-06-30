import argparse
import os
import wfdb
import natsort  # module used to sort file names
from concurrent.futures import ProcessPoolExecutor

from signal_api import extract_beats
from utils import files_in_dir, remove_extension
from constants import SIGNAL_DIR


def process_signal(signal_path):
    # get annotation data frame of signal file
    ann = wfdb.rdann(signal_path, 'atr', return_label_elements=[
        'symbol', 'description', 'label_store'], summarize_labels=True)
    # uncomment to save images of beats
    extract_beats(signal_path, ann)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--cpu', type=int, default=12,
                      help='number of cpu to use')

    args = args.parse_args()
    cpu = args.cpu

    # get all .hea and .dat files (respectively)
    signal_files = files_in_dir('.hea', SIGNAL_DIR)

    # sort file names in ascending order in list
    signal_files = natsort.natsorted(signal_files)

    # extract and save beats from file provided
    with ProcessPoolExecutor(cpu) as executor:
        for signal_file in signal_files:
            signal_path = os.path.join(
                SIGNAL_DIR,
                remove_extension(signal_file)
            )
            # get annotation data frame of signal file
            executor.submit(process_signal, signal_path)
