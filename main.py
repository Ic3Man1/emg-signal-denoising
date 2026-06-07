import wfdb

from noise import create_noisy_data
from data_operations import compare_all_methods

RECORD = wfdb.rdrecord("data_set/session3_participant1_gesture10_trial1/session3_participant1_gesture10_trial1")
SIGNAL = RECORD.p_signal

EMG_INDICIES = [i for i, name in enumerate(RECORD.sig_name) if not('U' in name.upper())]
DATA = SIGNAL[:, EMG_INDICIES]

if __name__ == "__main__":
    noisy_data = create_noisy_data(DATA, snr_db=10)
    compare_all_methods(DATA, noisy_data, channel_idx=0)