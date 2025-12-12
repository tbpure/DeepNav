from pathlib import Path

import numpy as np

from ekf import geoparams
from ekf.read_data_from_log import read_gps_data

path = Path('/Users/yangyu/PycharmProjects/DeepNav/DeepNav_data/flight_csvs/0000_5.02')
timestamp, gps_lla, eph, epv, vel_lla = read_gps_data(path)
gps_ecef = geoparams.lla2ecef_batch(gps_lla, is_degree=True)
vel_ecef = np.diff(gps_ecef, axis=0)
pass