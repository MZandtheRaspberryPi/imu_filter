import math
import numpy as np
from typing import List
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

"""A python script to plot the state of a differential drive robot using a variety of filters."""

import os
import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy

Q = 0.01 * np.eye(9)
R = 0.1 * np.array([[1]])

INITIAL_ESTIMATE = np.zeros((9,1))

INITIAL_COVARIANCE = 0.1 * np.eye(9)

THIS_FILE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(THIS_FILE_DIR, "..", "RepoIMU")
DATASET_PATH = os.path.join(DATASET_DIR, "TStick", "TStick_Test02_Trial1.csv")

def get_gaussian_noise(
        num_samples: int, mu: Any, sigma: Any) -> Any:
    """returns a multivariate gaussian noise for given mu array (nx1), and given sigma nxn covariance matrix

    Args:
        num_samples (int): how many samples to return
        mu (Any): expected value for each variable
        sigma (Any): covariance matrix for variable

    Returns:
        Any: num_samples x n numpy array with samples
    """
    num_gaussians = mu.shape[0]
    samples = np.zeros((num_samples, num_gaussians, 1))
    for i in range(0, num_gaussians):
        samples[:, i, 0] = np.random.normal(
            mu[i, 0], math.sqrt(sigma[i][i]), num_samples)
    return samples

def get_gaussian_error_ellipses(expected_value: Any, sigma: Any,
                                p_value: float = 0.95,
                                step_value: float = 0.01) -> Any:
    """function to generate plots of gaussian error ellipses

    Args:
        expected_value (Any): numpy nx1 array with means of each gaussian.
        sigma (Any): numpy nxn array covariance matrix
        p_value (float): p value for the error ellipses
        step_value (float, optional): when tracing the ellipse, we step over theta
            values and calculate the ellipse value at that theta.
            how granular should we step? Defaults to 0.01.

    Returns:
        Any: an Nx2 array, where arr[:, 0] is the x and arr[:, 1] is the y.
            N is determined by step size
            and will be equivalent to 2pi/step_size.
    """
    # comes from solving gaussian pdf integral for the quadratic corresponding to ellipse
    r = math.sqrt(-2 * math.log((1-p_value)))

    ellipses_x = np.arange(0, 2 * math.pi, step_value)
    ellipses_y = np.arange(0, 2 * math.pi, step_value)
    ellipses_x = r * np.cos(ellipses_x)
    ellipses_y = r * np.sin(ellipses_y)

    ellipse_normal = np.zeros((len(ellipses_x), 2))
    ellipse_normal[:, 0] = ellipses_x
    ellipse_normal[:, 1] = ellipses_y
    ellipse_transformed = np.matmul(
        ellipse_normal, scipy.linalg.sqrtm(sigma)) + expected_value

    return ellipse_transformed

def load_dataset(dataset_path: str):

    df = pd.read_csv(dataset_path,
                     header=[0,1], delimiter=";")
    df = df.iloc[:, :-1]

    df.columns = ["time_s", "vicon_orientation_w", "vicon_orientation_x", "vicon_orientation_y", "vicon_orientation_z",
                  "imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z", "imu_magn_x",
                  "imu_magn_y", "imu_magn_z"]

    quat_cols = df.loc[:, ["vicon_orientation_w", "vicon_orientation_x", "vicon_orientation_y", "vicon_orientation_z"]].values

    qw, qx, qy, qz = quat_cols[:, 0] , quat_cols[:, 1], quat_cols[:, 2], quat_cols[:, 3]

    df["orientation_x"] = np.arctan2(2*(qw * qx + qy * qz), ((1 - 2*(qy**2 + qz**2))))
    df["orientation_y"] = -math.pi / 2 + 2 * np.arctan2(np.sqrt(1+2*(qw * qy - qx * qz)), np.sqrt(1 - 2*(qw*qy + qx*qz)))
    df["orientation_z"] = np.arctan2(2*(qw * qz + qx * qy), ((1 - 2*(qy**2 + qz**2))))
    return df

def get_a_matrix(
        prior_state: Any, delta_t: float, angular_rotation: Any,
        gravitational_constant: float = -9.81):
    s_x, s_y, s_z = prior_state[0:3, 0]
    v_x, v_y, v_z = prior_state[3:6, 0]
    phi, theta, psi = prior_state[6:, 0]

    omega_x, omega_y, omega_z = angular_rotation[:, 0]

    a_matrix = np.zeros((prior_state.shape[0], prior_state.shape[0]))

    a_matrix[0,0] = 1
    a_matrix[0,1] = 0
    a_matrix[0,2] = 0
    a_matrix[0,3] = delta_t * math.cos(theta) * math.cos(psi) 
    a_matrix[0,4] = delta_t * math.cos(theta) * math.sin(psi) 
    a_matrix[0,5] = delta_t*math.sin(theta)
    a_matrix[0,6] =  0 
    a_matrix[0,7] = (v_x * (-math.sin(theta)) * math.cos(psi) + v_y *(-math.sin(theta)) * math.sin(psi) - v_z * math.cos(theta)) * delta_t
    a_matrix[0,8] = delta_t * (v_x * math.cos(theta) * (-math.sin(psi )) + v_y * math.cos(theta) * math.cos(psi))
    a_matrix[1,0] = 0
    a_matrix[1,1] = 1
    a_matrix[1,2] = 0
    a_matrix[1,3] = delta_t * ( math.sin(phi)* math.sin(theta) * math.cos(psi) - math.cos(phi) * math.sin(psi)  )
    a_matrix[1,4] = delta_t * (math.sin(phi) * math.sin(theta) * math.sin(psi) + math.cos(phi) * math.cos(psi) )
    a_matrix[1,5] =  delta_t * (math.sin(phi) * math.cos(theta))
    a_matrix[1,6] =  delta_t * (v_x * (math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)) + v_y * (math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)) + v_z * (math.cos(phi) * math.cos(theta)))
    a_matrix[1,7] =   delta_t * (v_x * (math.cos(phi) * math.cos(theta) * math.cos(psi)) + v_y * (math.sin(phi) * math.cos(theta) * math.sin(psi)) + v_z * (math.sin(phi) * ( -math.sin(theta))))
    a_matrix[1,8] = delta_t * (v_x * (math.sin(phi) * math.sin(theta) * (-math.sin(psi))) + v_y * (math.sin(phi) * math.sin(theta) * math.cos(psi) + math.cos(phi) * (-math.sin(psi))))
    a_matrix[2,0] = 0
    a_matrix[2,1] = 0
    a_matrix[2,2] = 1
    a_matrix[2,3] = delta_t * (math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi))
    a_matrix[2,4] = delta_t * (math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi))
    a_matrix[2,5] =  delta_t * (math.cos(phi) * math.cos(theta))
    a_matrix[2,6] =  delta_t * (v_x * ((-math.sin(phi)) * math.sin(theta) * math.cos(psi) + math.cos(phi) * math.sin(psi)) + v_y*((-math.sin(phi)) * math.sin(theta) * math.sin(psi) - math.cos(phi) * math.cos(psi) ) + v_z * ((-math.sin(phi)) * math.cos(theta)))
    a_matrix[2,7] =   delta_t * (v_x * (math.cos(phi) * math.cos(theta) * math.cos(psi)) + v_y * (math.cos(phi) * math.cos(theta) * math.sin(psi) ) + v_z * ((math.cos(phi) * (-math.sin(theta)))))
    a_matrix[2,8] = delta_t * (v_x * (math.cos(phi) * math.sin(theta) * (-math.sin(psi)) + math.sin(phi) * math.cos(psi) ) + v_y * (math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)))

    a_matrix[3, 0] = 0
    a_matrix[3, 1] = 0
    a_matrix[3, 2] = 0
    a_matrix[3, 3] = 1
    a_matrix[3, 4] = delta_t * (-omega_z)
    a_matrix[3, 5] = delta_t * (omega_y)
    a_matrix[3, 6] = 0
    a_matrix[3, 7] = delta_t * (-gravitational_constant * math.cos(theta))
    a_matrix[3, 8] = 0
    a_matrix[4, 0] = 0
    a_matrix[4, 1] = 0
    a_matrix[4, 2] = 0
    a_matrix[4, 3] = delta_t * (omega_z)
    a_matrix[4, 4] = 1
    a_matrix[4, 5] = delta_t * (-omega_x)
    a_matrix[4, 6] = delta_t * (gravitational_constant * math.cos(phi) * math.cos(theta))
    a_matrix[4, 7] = delta_t * (gravitational_constant * math.sin(phi) * (-math.sin(theta)))
    a_matrix[4, 8] = 0
    a_matrix[5, 0] = 0
    a_matrix[5, 1] = 0
    a_matrix[5, 2] = 0
    a_matrix[5, 3] = delta_t * (-omega_y)
    a_matrix[5, 4] = delta_t * (omega_x)
    a_matrix[5, 5] = 1
    a_matrix[5, 6] = delta_t * (gravitational_constant * (-math.sin(phi)) * math.cos(theta))
    a_matrix[5, 7] = delta_t * (gravitational_constant * math.cos(phi) * (-math.sin(theta)))
    a_matrix[5, 8] = 0
    a_matrix[6, 0] = 0
    a_matrix[6, 1] = 0
    a_matrix[6, 2] = 0
    a_matrix[6, 3] = 0
    a_matrix[6, 4] = 0
    a_matrix[6, 5] = 0
    a_matrix[6, 6] = 1 + math.cos(phi) * math.tan(theta) * omega_y - math.sin(phi) * math.tan(theta) * omega_z
    a_matrix[6, 7] = math.sin(phi) * ((1/math.cos(theta)) ** 2) * omega_y + math.cos(phi) * ((1/math.cos(theta)) ** 2) * omega_z
    a_matrix[6, 8] = 0
    a_matrix[7, 0] = 0
    a_matrix[7, 1] = 0
    a_matrix[7, 2] = 0
    a_matrix[7, 3] = 0
    a_matrix[7, 4] = 0
    a_matrix[7, 5] = 0
    a_matrix[7, 6] = -math.sin(phi) * omega_y - math.cos(phi) * math.cos(omega_z)
    a_matrix[7, 7] = 1
    a_matrix[7, 8] = 0
    a_matrix[8, 0] = 0
    a_matrix[8, 1] = 0
    a_matrix[8, 2] = 0
    a_matrix[8, 3] = 0
    a_matrix[8, 4] = 0
    a_matrix[8, 5] = 0
    a_matrix[8, 6] = (math.cos(phi) / math.cos(theta)) * omega_y + (-math.sin(phi) / math.cos(theta)) * omega_z
    a_matrix[8, 7] = math.sin(phi) * omega_y * (-1 / (math.cos(theta)) ** 2) * (-math.sin(theta)) + math.cos(
        phi) * omega_z * (-1 / (math.cos(theta) ** 2)) * (-math.sin(theta))
    a_matrix[8, 8] = 1

    return a_matrix


def get_c_matrix(state_t_given_t_minus_one: Any, magnitational_vector: Any):

    s_x, s_y, s_z = state_t_given_t_minus_one[0:3, 0]
    v_x, v_y, v_z = state_t_given_t_minus_one[3:6, 0]
    phi, theta, psi = state_t_given_t_minus_one[6:, 0]

    a, _, b = magnitational_vector[:, 0]

    c_matrix = np.zeros((3,9))

    """

    c_matrix[0, 0] = 0
    c_matrix[0, 1] = 0
    c_matrix[0, 2] = 0
    c_matrix[0, 3] = 0
    c_matrix[0, 4] = 0
    c_matrix[0, 5] = 0
    c_matrix[0, 6] = 1
    c_matrix[0, 7] = 0
    c_matrix[0, 8] = 0
    c_matrix[1, 0] = 0
    c_matrix[1, 1] = 0
    c_matrix[1, 2] = 0
    c_matrix[1, 3] = 0
    c_matrix[1, 4] = 0
    c_matrix[1, 5] = 0
    c_matrix[1, 6] = 2 * np.cos(phi) * np.cos(theta) / (np.cos(2 * phi) - np.cos(2 * theta) - 2)
    c_matrix[1, 7] = 2 * np.sin(phi) * np.sin(theta) / (np.cos(2 * phi) - np.cos(2 * theta) - 2)
    c_matrix[1, 8] = 0
    c_matrix[2, 0] = 0
    c_matrix[2, 1] = 0
    c_matrix[2, 2] = 0
    c_matrix[2, 3] = 0
    c_matrix[2, 4] = 0
    c_matrix[2, 5] = 0
    c_matrix[2, 6] = a * (1 / np.cos(phi)) ** 2 * (1 / np.sin(theta)) * (1 / np.cos(theta)) / (
                a ** 2 + np.tan(phi) ** 2 * (1 / np.sin(theta)) ** 2 * (1 / np.cos(theta)) ** 2)
    c_matrix[2, 7] = a * np.tan(phi) * np.cos(2 * theta) * (1 / np.sin(theta)) ** 2 * (
                1 / np.cos(theta)) ** 2 / (
                                 a ** 2 + np.tan(phi) ** 2 * (1 / np.sin(theta)) ** 2 * (1 / np.cos(theta)) ** 2)
    c_matrix[2, 8] = 0
    """
    c_matrix[0, 6] = 1
    c_matrix[1, 7] = 1
    c_matrix[2, 8] = 1

    return c_matrix


def get_next_true_state(prior_state: Any, delta_t: float, angular_rotation: Any,
                        noise: Any, accelerometer: Any,
                        gravitational_constant: float = -9.81):
    s_x, s_y, s_z = prior_state[0:3, 0]
    v_x, v_y, v_z = prior_state[3:6, 0]
    phi, theta, psi = prior_state[6:, 0]

    omega_x, omega_y, omega_z = angular_rotation[:, 0]

    a_x, a_y, a_z = accelerometer[:, 0]

    new_state = np.zeros((prior_state.shape[0], 1))
    # new_state[0,0] = s_x + (v_x * math.cos(theta) * math.cos(psi)  + v_y * math.cos(theta) * math.sin(psi) - v_z * math.sin(theta)) * delta_t
    # new_state[1,0] = s_y + (v_x * (math.cos(phi) * math.sin(theta) * math.cos(psi) - math.cos(phi) * math.sin(psi)) + v_y * (math.sin(phi) * math.sin(theta) * math.sin(psi) + math.cos(phi) * math.cos(psi)) + v_z * (math.sin(phi) * math.cos(theta)) ) * delta_t
    # new_state[2,0] = s_z + (v_x * (math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)) + v_y * (math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)) + v_z * (math.cos(phi) * math.cos(theta)) ) * delta_t
    # new_state[3, 0] = v_x + delta_t * (a_x - gravitational_constant * math.sin(theta)  + omega_y * v_z - omega_z * v_y)
    # new_state[4, 0] = v_y + delta_t * (a_y + gravitational_constant * math.sin(phi) * math.cos(theta) - omega_x * v_z + omega_z * v_x)
    # new_state[5, 0] = v_z + delta_t * (a_z + gravitational_constant * math.cos(phi) * math.cos(theta) + omega_x * v_y - omega_y * v_x )
    new_state[6, 0] = phi + omega_x * delta_t #+ math.sin(phi) * math.tan(omega_y) + math.cos(phi) * math.tan(theta) * omega_z
    new_state[7, 0] = theta + omega_y * delta_t #math.cos(phi) * omega_y - math.sin(phi) * omega_z
    new_state[8, 0] = psi + omega_z * delta_t #(math.sin(phi) / np.cos(theta)) * omega_y + (math.cos(phi) / np.cos(theta)) * omega_z

    return new_state + noise

def calc_digital_compass2(state_estimate: Any, gravity, magnitational_vector):

    phi, theta, psi = state_estimate[6:, 0]

    a, _, b = magnitational_vector[:, 0]

    digital_compass = np.zeros((3,1))
    digital_compass[0,0] = math.atan2((-gravity * math.sin(phi) * math.cos(theta)), (-gravity * math.cos(phi) * math.cos(theta)))
    digital_compass[1, 0] = math.atan2((gravity * math.sin(theta)),
                                       (-gravity * math.sin(phi) * math.cos(theta) * math.sin(phi) - gravity * math.cos(phi) * math.cos(theta) * math.cos(phi)))
    digital_compass[2, 0] = math.atan2((-math.sin(phi)), (a * math.cos(theta) * math.sin(theta) * math.cos(phi) * b))
    return digital_compass


def calc_digital_compass(accelerometer: Any, magnetometer: Any, gravity: float, magnitational_vector: Any):

    """
    phi, theta, psi = state_estimate[6:, 0]

    a, _, b = magnitational_vector[:, 0]

    digital_compass = np.zeros((3,1))
    digital_compass[0,0] = math.atan2((-gravity * math.sin(phi) * math.cos(theta)), (-gravity * math.cos(phi) * math.cos(theta)))
    digital_compass[1, 0] = math.atan2((gravity * math.sin(theta)),
                                       (-gravity * math.sin(phi) * math.cos(theta) * math.sin(phi) - gravity * math.cos(phi) * math.cos(theta) * math.cos(phi)))
    digital_compass[2, 0] = math.atan2((-math.sin(phi)), (a * math.cos(theta) * math.sin(theta) * math.cos(phi) * b))
    """
    g_b_x, g_b_y, g_b_z = accelerometer[:, 0]

    phi = math.atan2(g_b_y, g_b_z)
    theta = math.atan2(-g_b_x, (g_b_y * math.sin(phi) + g_b_z * math.cos(phi)))

    x_rotation = np.array([[1, 0, 0],
                           [0, math.cos(phi), -math.sin(phi)],
                            [0, math.sin(phi), math.cos(phi)]])

    y_rotation = np.array([[math.cos(theta), 0, math.sin(theta)],
                           [0, 1, 0],
                           [-math.sin(theta), 0, math.cos(theta)]])

    h_hor = y_rotation @ (x_rotation @ magnitational_vector)

    h_hor_y = h_hor[1, 0]
    h_hor_x = h_hor[0, 0]

    psi = math.atan2(h_hor_y, h_hor_x)

    orientation = np.zeros((3,1))
    orientation[0,0] = phi
    orientation[1,0] = theta
    orientation[2,0] = psi

    return orientation


def get_mu_sigma_extended_kalman(
        initial_state: Any, initial_covariance: Any, Q: Any,
        R: Any, process_noise_series: Any, gyroscope_data: Any, accelerometer_data: Any, time_step_data: Any, gravity: float,
        magnetic_vector: Any):

    time_steps = gyroscope_data.shape[0]
    mu_estimates = np.zeros((time_steps + 1, initial_state.shape[0], 1))
    mu_estimates[0, :, :] = initial_state

    sigma_estimates = np.zeros(
        (time_steps + 1, initial_state.shape[0],
         initial_state.shape[0]))
    sigma_estimates[0, :, :] = initial_covariance

    prior_time_seconds = 0

    total_time = 0.0
    for i in range(1, accelerometer_data.shape[0] + 1):
        start_time = time.time()
        prior_estimate = mu_estimates[i - 1, :, :]
        sigma_t_minus_one_given_t_minus_one = sigma_estimates[i - 1, :, :]

        accel_measurement = accelerometer_data[i - 1, :].reshape((3,1))
        gyro_measurement = gyroscope_data[i - 1, :].reshape((3,1))
        delta_t = time_step_data[i - 1, 0] - prior_time_seconds
        prior_time_seconds = time_step_data[i - 1, 0]
        process_noise = process_noise_series[ i -1, :, :]

        mu_t_given_t_minus_one = get_next_true_state(prior_estimate, delta_t, gyro_measurement,
            process_noise, accel_measurement,
            gravity)

        a_matrix = get_a_matrix(prior_estimate, delta_t, gyro_measurement,
            gravity)

        sigma_t_given_t_minus_one = np.matmul(a_matrix, np.matmul(
            sigma_t_minus_one_given_t_minus_one, a_matrix.transpose())) + Q

        measurement = calc_digital_compass(accel_measurement, np.zeros((3,1)), -9.81, magnetic_vector)

        error_vs_estimate = measurement - mu_t_given_t_minus_one[6:, :]

        c_matrix = get_c_matrix(mu_t_given_t_minus_one, magnetic_vector)

        sigma_ct = np.matmul(
            sigma_t_given_t_minus_one, c_matrix.transpose())
        c_sigma_ct = np.matmul(c_matrix, sigma_ct)
        c_sigma_ct_plus_R = c_sigma_ct + R
        error_scaling_paranthesis = np.linalg.inv(c_sigma_ct_plus_R)
        error_scaling = np.matmul(
            sigma_t_given_t_minus_one, np.matmul(
                c_matrix.transpose(),
                error_scaling_paranthesis))
        mu_t_given_t = mu_t_given_t_minus_one + \
                       np.matmul(error_scaling, error_vs_estimate)
        mu_estimates[i, :, :] = mu_t_given_t

        cov_update = sigma_t_given_t_minus_one - \
                     np.matmul(error_scaling, np.matmul(
                         c_matrix, sigma_t_given_t_minus_one))
        sigma_estimates[i, :, :] = cov_update

        total_time += time.time() - start_time

    return mu_estimates, sigma_estimates, total_time / time_steps

def main():
    ground_truth = load_dataset(DATASET_PATH)

    process_noise = get_gaussian_noise(
        ground_truth.shape[0], np.zeros((INITIAL_ESTIMATE.shape[0], 1)),
        Q)

    a_matrix = get_a_matrix(INITIAL_ESTIMATE, 0.01, np.zeros((3,1)))
    
    # c_matrix = get_c_matrix(INITIAL_ESTIMATE, np.zeros((3,1)))

    next_state = get_next_true_state(INITIAL_ESTIMATE, 0.01, np.zeros((3,1)),
                        np.zeros((9,1)), np.zeros((3,1)),
                        gravitational_constant = -9.81)

    magnetic_field = np.zeros((3,1))

    """
     magnetic north vector mg = [cos(ϕ
    L) − sin(ϕ
    L)]T
    , where ϕ
    L
    is the geographical latitude angle. For the geographical position of the laboratory (geographic coordinates: 50.35363, 18.9148285), where measurements were
    done, we have ϕ
    L = 66o = 1.1519 rad.
    """
    # x, y, z, for Shelton, WA
    magnetic_field[0,0] = math.cos(1.1519)
    magnetic_field[1,0] = 0
    magnetic_field[2,0] = -math.sin(1.1519)

    mu_estimates, sigma_estimates, run_times = get_mu_sigma_extended_kalman(INITIAL_ESTIMATE, INITIAL_COVARIANCE,
                                        Q, R, process_noise, ground_truth[["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]].values[:, :], ground_truth[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].values[:, :], ground_truth[["time_s"]].values[:, :], -9.81,
       magnetic_field)

    orientation_measurements_digital_compass = np.zeros((ground_truth.shape[0], 3, 1))
    orientation_measurements_gyro = np.zeros((ground_truth.shape[0], 3, 1))
    orientation_measurements_true_state = np.zeros((ground_truth.shape[0], 3, 1))

    orientation_gyro = np.zeros((3,1))
    prior_state = np.zeros((9,1))
    for i in range(ground_truth.shape[0]):
        accel_measurement = ground_truth[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].values[i, :].reshape((3,1))
        gyro_measurement = ground_truth[["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]].values[i, :].reshape((3,1))
        magneto_measurement = ground_truth[["imu_magn_x", "imu_magn_y", "imu_magn_z"]].values[i, :].reshape((3,1))
        orientation = calc_digital_compass(accel_measurement, np.zeros((3,1)), -9.81, magnetic_field)

        next_state = get_next_true_state(prior_state, 0.01, gyro_measurement,
            np.zeros((9,1)), accel_measurement,
            -9.81)

        prior_state = next_state

        orientation_gyro = orientation_gyro + magneto_measurement

        orientation_measurements_digital_compass[i, :, :] = orientation
        orientation_measurements_gyro[i, :, :] = orientation_gyro
        orientation_measurements_true_state[i, :, :] = next_state[6:, :]

    orientation_measurements_true_state = orientation_measurements_true_state * 180 / np.pi
    orientation_measurements_true_state = np.mod(orientation_measurements_true_state, 360)

    estimates_to_plot = 8500

    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Ground Truth, {} from imu_repo".format(os.path.basename(DATASET_PATH)))
    ax[0].plot(ground_truth["time_s"], ground_truth["orientation_x"], label="true_orientation_x_radians")
    #ax[0].plot(ground_truth["time_s"], orientation_measurements_digital_compass[:, 0, 0], label="compass_orientation_x_radians")
    ax[0].plot(ground_truth["time_s"].values[:estimates_to_plot], mu_estimates[1:estimates_to_plot+1, 6, 0], label="estimate_orientation_x_radians")
    ax[1].plot(ground_truth["time_s"], ground_truth["orientation_y"], label="true_orientation_y_radians")
    #ax[1].plot(ground_truth["time_s"], orientation_measurements_digital_compass[:, 1, 0], label="compass_orientation_y_radians")
    ax[1].plot(ground_truth["time_s"].values[:estimates_to_plot], mu_estimates[1:estimates_to_plot+1, 7, 0], label="estimate_orientation_y_radians")
    ax[2].plot(ground_truth["time_s"], ground_truth["orientation_z"], label="true_orientation_z_radians")
    #ax[2].plot(ground_truth["time_s"], orientation_measurements_digital_compass[:, 2, 0], label="compass_orientation_z_radians")
    ax[2].plot(ground_truth["time_s"].values[:estimates_to_plot], mu_estimates[1:estimates_to_plot+1, 8, 0], label="estimate_orientation_z_radians")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(ground_truth["time_s"].values, sigma_estimates[1:, 6, 6])
    ax.set_xlabel("time")
    ax.set_ylabel("variance")
    fig.suptitle("Variance of X Angle Estimate Over Time")
    plt.show()


    # graph digital compass vs orientation


if __name__ == "__main__":
    main()
