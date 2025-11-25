#!/usr/bin/env python3
"""
Created on Mon 09 Nov 2020 | 5:25 PM

@author: Ahmed Majuid

Usage:
Define the network architecture and training hyperparameters
"""

import os
from openpyxl import load_workbook
import pandas as pd
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf messages
import tensorflow as tf
import training
import utils
import postprocessing
from preprocessing.create_dataset import create_dataset

print(tf.config.list_physical_devices('GPU'))
# Session Parameters
session_mode = ["Fresh", "Resume", "Evaluate", "Override"]
mode_id = 0
gpu_name = ["/GPU:0", "/GPU:1", None]
gpu_id = 0
create_new_dataset = True

# Network Architecture
model_architecture = [
    # 1D 卷积层，用于提取局部时间特征
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
                           input_shape=(None, 10)),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    # tf.keras.layers.MaxPooling1D(pool_size=2),  # 可选，缩短序列长度
    tf.keras.layers.GRU(20, return_sequences=True), #200
    tf.keras.layers.GRU(200, return_sequences=True),
    tf.keras.layers.GRU(200, return_sequences=True),
    tf.keras.layers.GRU(20, return_sequences=False), #200S
    tf.keras.layers.Dense(6)
    ]

# looping on parameters
varying_hyperparam = None
hyperparam_values = [None]
result_path = Path("DeepNav_results")
folder_count = sum(1 for p in result_path.iterdir() if p.is_dir())
existed_files = []
existed_files = [p.name for p in result_path.iterdir() if p.is_dir()]
existed_files.sort()
file_index = int(existed_files[-1][-3:]) + 1

# Network Hyperparameters
session_data = {"trial_number" : file_index,
                'model_architecture' : 'GRU',
                "session_mode" : session_mode[mode_id],
                "gpu_name" : gpu_name[gpu_id],

                "batch_size" : int(1 * 1024),
                "learning_rate" : 0.001,
                "window_size" : 100, # 200
                "dropout" : 0.0,
                "epochs" : 100,  # 100
                "initial_epoch" : 0
                }

excel_path = 'DeepNav_results/train_records.csv'

df = pd.DataFrame([session_data])

csv_path = 'DeepNav_results/train_records.csv'
try:
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
except Exception as e:
    print(f"写入 CSV 文件时出错: {e}")

# 打开设备日志，训练时会显示每个操作在哪个设备上运行
tf.debugging.set_log_device_placement(True)


# 创建文件夹
# create folders for the training outputs (weights, plots, loss history)
trial_tree = utils.create_trial_tree(session_data["trial_number"], session_data["session_mode"])

if create_new_dataset:
    session_data["dataset_name"] = None
    # 输入特征：三轴角速度、三轴加速度、三轴磁力数据
    colum_names = {"features"     : ["w_x", "w_y", "w_z", "a_x", "a_y", "a_z", "m_x", "m_y", "m_z"],
                    "features_diff": ["h"],
                    "labels"       : ["Vn", "Ve", "Vd", "Pn", "Pe", "Pd"]}
else:
    session_data["dataset_name"] = "T001_logs548_F10L6_W50_03Dec2020_1542_FMUV5"
    colum_names = {}

# create windowed datasets from the flight csv files (or retrieve an old one from binary files)
train_ds, val_dataset, train_flights_dict, val_flights_dict, signals_weights = create_dataset(session_data, colum_names)

# batch and shuffle
train_dataset = train_ds.batch(session_data["batch_size"]).shuffle(buffer_size=1000)
val_dataset = val_dataset.batch(session_data["batch_size"]).shuffle(buffer_size=1000)

# print the shape of a single batch
for x, y in train_dataset.take(1):
    print("\nshape of a single training batch")
    print(x.shape, y.shape)

# convert signals weights to a tensor to be used by the loss function
signals_weights_tensor = tf.constant(signals_weights, dtype=tf.float32)
if tf.config.list_physical_devices('GPU'):
    print("training on GPU 😊😊😊😊")
else :
    print("training on CPU 😠😠😠😠")

# start training
with tf.device('/GPU:0'):
    model = training.start_training(session_data, model_architecture, train_dataset, val_dataset, \
                                signals_weights_tensor, trial_tree)

# for every flight, plot all states (truth vs predictions)
flights_summary = postprocessing.evaluate_all_flights(model, train_flights_dict, val_flights_dict, \
                                    trial_tree["trial_root_folder"], n_extreme_flights=30)

# add the network configuration and performance to the summary csv
postprocessing.summarize_session(trial_tree, model, session_data, flights_summary)

# save a keras model
keras_model_path = trial_tree["trial_root_folder"] + "/keras_model"
model.save(keras_model_path)

# save the model in tf SavedModel format
tf_model_path = trial_tree["trial_root_folder"] + "/tf_saved_model"
tf.saved_model.save(model, tf_model_path)