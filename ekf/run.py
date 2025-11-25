from ekf.read_data_from_log import read_mag_data, read_sensor_combined, read_gps_data, read_estimator_results
from loose_ekf import LooseEKF

def run(file_path):
    ekf = LooseEKF(dt=0.01)
    timestamp_mag, mag = read_mag_data(file_path=file_path)
    timestamp_sensors, gyro, acc = read_sensor_combined(file_path=file_path)
    timestamp_gps, gps, eph, epv, vel = read_gps_data(file_path=file_path)
    timestamp_estimator, quat, vel, position, covariances = read_estimator_results(file_path=file_path)

    index_gps, index_estimator, index_mag = 0

    for i in range(len(timestamp_sensors - 1)):
        t_gps = timestamp_gps[index_gps]
        t_mag = timestamp_mag[index_mag]
        t_est = timestamp_estimator[index_estimator]
        t_now = timestamp_sensors[i]



