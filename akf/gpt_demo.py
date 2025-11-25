# Python demo: Adaptive Kalman Filter (AKF) for GNSS/INS loose coupling
# - 2D motion (x, y), state: [x, y, vx, vy]
# - IMU provides acceleration (ax, ay) at 100 Hz (no attitude dynamics for simplicity)
# - GNSS provides position at 1 Hz
# - Loosely coupled: GNSS measurement updates position only
# - Adaptation: online estimate of measurement noise R using innovation statistics (exponential window)
# Run this cell to simulate and plot results.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulation params
sim_time = 60.0        # seconds
imu_hz = 100.0
gnss_hz = 1.0
dt = 1.0 / imu_hz
steps = int(sim_time / dt)

# True motion: constant velocity with a small maneuver in the middle
true_v = np.array([2.0, 0.5])  # m/s
true_pos = np.array([0.0, 0.0])

# IMU noise (accelerometer)
accel_sigma = 0.05  # m/s^2 (std)
# GNSS noise (position)
gnss_sigma_init = 2.0  # m (initial assumed)
# Process noise for KF (modeling mismatch)
process_accel_noise = 0.02  # m/s^2

# Measurement adaptation params
adapt_alpha = 0.98  # forgetting factor for R estimation

# State: x = [px, py, vx, vy]
x = np.zeros(4)
P = np.diag([10.0, 10.0, 1.0, 1.0])  # initial covariance

# State transition using continuous-time acceleration input u = [ax, ay]
def predict_state(x, u, dt):
    # simple constant-accel motion model integration
    px, py, vx, vy = x
    ax, ay = u
    px += vx * dt + 0.5 * ax * dt * dt
    py += vy * dt + 0.5 * ay * dt * dt
    vx += ax * dt
    vy += ay * dt
    return np.array([px, py, vx, vy])

# Linearized F and B for discrete time (constant accel input)
def get_F_B(dt):
    F = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1,0],
                  [0,0,0,1]])
    B = np.array([[0.5*dt*dt, 0],
                  [0, 0.5*dt*dt],
                  [dt, 0],
                  [0, dt]])
    return F, B

F, B = get_F_B(dt)

# Process noise covariance Q (from accel noise)
Qa = (process_accel_noise**2) * np.eye(2)
Q = B @ Qa @ B.T

# Measurement matrix H (position only)
H = np.array([[1,0,0,0],
              [0,1,0,0]])

# Storage for plotting
true_positions = []
est_positions = []
gnss_positions = []
time_array = []
innovations = []         # innovation vector (2D) at each GNSS update
innov_var_est = []       # estimated measurement covariance trace (R)
R_est = (gnss_sigma_init**2) * np.eye(2)  # initial R estimate

# simulation loop: IMU at 100Hz, GNSS at 1Hz
gnss_interval = int(imu_hz / gnss_hz)
step = 0
for k in range(steps):
    t = k * dt
    # Introduce a small acceleration maneuver between 20s and 30s
    if 20.0 <= t < 30.0:
        maneuver_acc = np.array([0.5, -0.2])
    else:
        maneuver_acc = np.array([0.0, 0.0])
    # True acceleration equals maneuver (no gravity in 2D plane)
    true_acc = maneuver_acc
    # IMU measures acceleration with noise (assume bias-free for demo)
    imu_meas = true_acc + np.random.randn(2) * accel_sigma

    # True state propagation (for generating ground truth)
    true_pos = true_pos + true_v * dt + 0.5 * true_acc * dt * dt
    true_v = true_v + true_acc * dt

    # KF prediction using IMU measurement as control input
    x = predict_state(x, imu_meas, dt)
    # Covariance propagation
    F, B = get_F_B(dt)
    P = F @ P @ F.T + Q

    # GNSS update at lower rate
    if (k % gnss_interval) == 0:
        # GNSS measurement (position) with noise
        gnss_meas = true_pos + np.random.randn(2) * gnss_sigma_init

        # Innovation
        z_pred = H @ x
        innov = gnss_meas - z_pred
        S = H @ P @ H.T + R_est  # innovation covariance using current R estimate

        # Kalman gain and update
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ innov
        P = (np.eye(4) - K @ H) @ P

        # Adapt measurement covariance R using exponential forgetting and innovation
        # R_new = alpha * R_old + (1 - alpha) * (innov * innov^T - H P H^T)
        # To ensure positive-definiteness, we take symmetric part and clip eigenvalues
        sample_R = np.outer(innov, innov) - H @ P @ H.T
        R_est = adapt_alpha * R_est + (1 - adapt_alpha) * sample_R
        # Symmetrize
        R_est = 0.5 * (R_est + R_est.T)
        # Regularize: enforce minimum variance
        min_var = 0.1**2
        vals, vecs = np.linalg.eigh(R_est)
        vals = np.clip(vals, min_var, 1e3)
        R_est = vecs @ np.diag(vals) @ vecs.T

        # store for plotting
        innovations.append(innov.copy())
        innov_var_est.append(np.trace(R_est))
        gnss_positions.append(gnss_meas.copy())

    true_positions.append(true_pos.copy())
    est_positions.append(x[:2].copy())
    time_array.append(t)

# Convert to arrays
true_positions = np.array(true_positions)
est_positions = np.array(est_positions)
gnss_positions = np.array(gnss_positions)
innovations = np.array(innovations)
innov_var_est = np.array(innov_var_est)

# Plot trajectories
plt.figure(figsize=(10,5))
plt.plot(true_positions[:,0], true_positions[:,1], label='True trajectory')
plt.plot(est_positions[:,0], est_positions[:,1], label='AKF estimated trajectory')
if len(gnss_positions)>0:
    plt.scatter(gnss_positions[:,0], gnss_positions[:,1], s=10, marker='x', label='GNSS measurements')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('GNSS/INS Loose Coupling - Adaptive KF demo')
plt.axis('equal')
plt.grid(True)
plt.show()

# Plot innovation components and estimated R trace
fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
if innovations.size > 0:
    ax[0].plot(np.linalg.norm(innovations,axis=1), label='Innovation norm')
ax[0].set_ylabel('innovation norm (m)')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(innov_var_est, label='trace(R_est)')
ax[1].set_ylabel('trace(R_est) (m^2)')
ax[1].set_xlabel('GNSS update index')
ax[1].legend()
ax[1].grid(True)
plt.show()