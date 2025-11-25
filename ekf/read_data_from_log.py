import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt


def read_sensor_combined(file_path):
    file_name = file_path.joinpath(f"{file_path.name}_sensor_combined_0.csv")
    data_frame = pd.read_csv(file_name).to_numpy()
    timestamp = data_frame[:, 0]
    gyro = data_frame[:, 1:4]
    acc = data_frame[:, 6:9]
    return timestamp, gyro, acc

def read_gps_data(file_path):
    file_name = file_path.joinpath(f"{file_path.name}_vehicle_gps_position_0.csv")
    data_frame = pd.read_csv(file_name).to_numpy()
    timestamp = data_frame[:, 0]
    gps = data_frame[:, 2:5]
    eph = data_frame[:, 8] # 水平定位精度因子
    epv = data_frame[:, 9] # 垂直定位精度因子
    vel = data_frame[:, 15:18] # [北向， 东向， 垂直(向下为正)]
    return timestamp, gps, eph, epv, vel

def read_mag_data(file_path):
    file_name = file_path.joinpath(f"{file_path.name}_vehicle_magnetometer_0.csv")
    data_frame = pd.read_csv(file_name).to_numpy()
    timestamp = data_frame[:, 0]
    return timestamp, data_frame[:, 1:4]

def read_estimator_results(file_path):
    file_name = file_path.joinpath(f"{file_path.name}_estimator_status_0.csv")
    data_frame = pd.read_csv(file_name)
    covariances = data_frame[[f'covariances[{i}]' for i in range(24)]].to_numpy()
    data_frame = data_frame.to_numpy()
    timestamp = data_frame[:, 0]
    quat = data_frame[:, 1:5]
    vel = data_frame[:, 5:8]
    position = data_frame[:, 9:11]
    return timestamp, quat, vel, position, covariances


if __name__ == '__main__':
    path = Path('/Users/yangyu/PycharmProjects/DeepNav/DeepNav_data/flight_csvs/0000_5.02')
    timestamp_sensors, _, _ = read_sensor_combined(path)
    timestamp_gps, _, _, _, _= read_gps_data(path)
    timestamp_mag, _,= read_mag_data(path)
    timestamp_estimator, _, _, _, covariance = read_estimator_results(path)
    print(len(timestamp_sensors))
    print(len(timestamp_gps))
    print(len(timestamp_mag))
    print(len(timestamp_estimator))


    time = np.arange(covariance.shape[0])  # x轴：时间步
    plt.figure(figsize=(15, 8))

    for i in range(24):
        plt.plot(time, covariance[:, i], label=f'cov[{i}]')

    plt.xlabel('Time step')
    plt.ylabel('Covariance value')
    plt.title('24-D Covariance Components over Time')
    plt.legend(ncol=4, fontsize=8)  # 调整图例布局
    plt.grid(True)
    plt.show()

