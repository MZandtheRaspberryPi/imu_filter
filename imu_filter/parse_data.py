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

Q = 0.01 * np.eye(3)
R = 0.1 * np.eye(4)

INITIAL_ESTIMATE = np.zeros((3,1))

INITIAL_COVARIANCE = 0.1 * np.eye(3)

THIS_FILE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(THIS_FILE_DIR, "..", "RepoIMU")
DATASET_PATH = os.path.join(DATASET_DIR, "TStick", "TStick_Test11_Trial2.csv")

def get_x_rotation(phi: float):

    x_rotation = np.array([[1, 0, 0],
                           [0, math.cos(phi), -math.sin(phi)],
                            [0, math.sin(phi), math.cos(phi)]])
    return x_rotation

def get_y_rotation(theta: float):

    y_rotation = np.array([[math.cos(theta), 0, math.sin(theta)],
                           [0, 1, 0],
                           [-math.sin(theta), 0, math.cos(theta)]])
    return y_rotation

def get_z_rotation(psi: float):
    z_rotation = np.array([[math.cos(psi), -math.sin(psi), 0],
                           [math.sin(psi), math.cos(psi), 0],
                           [0, 0, 1]])
    return z_rotation

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
                     header=None, delimiter=";",
                     skiprows=[0,1])

    df.columns = ["time_s", "vicon_orientation_w", "vicon_orientation_x", "vicon_orientation_y", "vicon_orientation_z",
                  "imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z", "imu_magn_x",
                  "imu_magn_y", "imu_magn_z"]

    quat_cols = df.loc[:, ["vicon_orientation_w", "vicon_orientation_x", "vicon_orientation_y", "vicon_orientation_z"]].values

    qw, qx, qy, qz = quat_cols[:, 0] , quat_cols[:, 1], quat_cols[:, 2], quat_cols[:, 3]


    df["orientation_x"] = np.arctan2(2*(qw * qx + qy * qz), (1 - 2*(qy*qy + qx*qx)))

    t_2 = 2 * (qw * qy - qz * qx)
    t_2 = np.where(t_2 > 1, 1, t_2)
    t_2 = np.where(t_2 < -1, -1, t_2)
    df["orientation_y"] = np.arcsin(t_2)

    t_3 = 2 * (qw * qz + qx * qy)
    t_4 = 1 - 2 * (qy * qy + qz * qz)
    df["orientation_z"] = np.arctan2(t_3, t_4)
    return df


def get_a_matrix_saito(
        prior_state: Any, delta_t: float, angular_rotation: Any,
        gravitational_constant: float = -9.81):

    phi, theta, psi = prior_state[:3, 0]

    omega_x, omega_y, omega_z = angular_rotation[:, 0]

    a_matrix = np.zeros((prior_state.shape[0], prior_state.shape[0]))

    a_matrix[0, 0] = delta_t * omega_y * np.cos(phi) * np.tan(theta) - delta_t * omega_z * np.sin(phi) * np.tan(theta) + 1
    a_matrix[0, 1] = delta_t * omega_y * np.sin(phi) * (np.tan(theta)**2 + 1) + delta_t * omega_z * np.cos(phi) * (np.tan(theta)**2 + 1)
    a_matrix[0, 2] = 0
    a_matrix[1, 0] = -delta_t * omega_y * np.sin(phi) - delta_t * omega_z * np.cos(phi)
    a_matrix[1, 1] = 1
    a_matrix[1, 2] = 0
    a_matrix[2, 0] = delta_t * omega_y * np.cos(phi) / np.cos(theta) - delta_t * omega_z * np.sin(phi) / np.cos(theta)
    a_matrix[2, 1] = delta_t * omega_y * np.sin(phi) * np.sin(theta) / (np.cos(theta) ** 2) + delta_t * omega_z * np.sin(theta) * np.cos(phi) / (np.cos(theta) ** 2)
    a_matrix[2, 2] = 1

    if np.isnan(a_matrix[2, 0]) or np.isinf(a_matrix[2, 0]):
        a_matrix[2, 0] = 0
    if np.isnan(a_matrix[2, 1]) or np.isinf(a_matrix[2, 1]):
        a_matrix[2, 0] = 0
    return a_matrix


def get_a_matrix(
        prior_state: Any, delta_t: float, angular_rotation: Any,
        gravitational_constant: float = -9.81):
    #s_x, s_y, s_z = prior_state[0:3, 0]
    #v_x, v_y, v_z = prior_state[3:6, 0]
    phi, theta, psi = prior_state[:3, 0]

    omega_x, omega_y, omega_z = angular_rotation[:, 0]

    a_matrix = np.zeros((prior_state.shape[0], prior_state.shape[0]))

    a_matrix[0,0] = 1 + delta_t * np.cos(phi) * np.tan(theta) * omega_y
    a_matrix[0,1] = delta_t * np.sin(phi) * (1/np.tan(theta))**2 * omega_y + delta_t * np.cos(phi) * (1 / np.tan(theta))**2 * omega_z
    if np.isnan(a_matrix[0,1]) or np.isinf(a_matrix[0,1]):
        a_matrix[0,1] = 0
    a_matrix[0,2] = 0
    a_matrix[1,0] = -delta_t * np.sin(phi)*omega_y - delta_t * np.cos(phi) * omega_z
    a_matrix[1,1] = 1
    a_matrix[1,2] = 0
    a_matrix[2,0] = delta_t * np.sin(phi) * omega_y * (-1/(np.cos(theta)**2))
    if np.isnan(a_matrix[2,0]) or np.isinf(a_matrix[2,0]):
        a_matrix[2,0] = 0

    a_matrix[2,1] = 0
    a_matrix[2,2] = 0

    return a_matrix


def get_c_matrix(state_t_given_t_minus_one: Any, magnitational_vector: Any, gravity: float = 9.81):

    #s_x, s_y, s_z = state_t_given_t_minus_one[0:3, 0]
    #v_x, v_y, v_z = state_t_given_t_minus_one[3:6, 0]
    phi, theta, psi = state_t_given_t_minus_one[:3, 0]

    a, _, b = magnitational_vector[:, 0]

    c_matrix = np.zeros((3,3))
    c_matrix[0,0] = 0
    c_matrix[0,1] = gravity * np.cos(theta)
    c_matrix[0,2] = 0
    c_matrix[1,0] = -gravity * np.cos(theta) * np.cos(phi)
    c_matrix[1,1] = gravity * np.sin(phi) * np.sin(theta)
    c_matrix[1,2] = 0
    c_matrix[2,0] = gravity * np.sin(phi) * np.cos(theta)
    c_matrix[2,1] = gravity * np.cos(phi) * np.sin(theta)
    c_matrix[2,2] = 0

    return c_matrix

def get_c_matrix_saito(state_t_given_t_minus_one: Any, magnitational_vector: Any, gravity: float = 9.81):
    phi, theta, psi = state_t_given_t_minus_one[:3, 0]

    c_matrix = np.zeros((4,3))
    c_matrix[0,0] = 0
    c_matrix[0,1] = 0
    c_matrix[0,2] = 1
    c_matrix[1,0] = 0
    c_matrix[1,1] = -gravity * np.cos(theta)
    c_matrix[1,2] = 0
    c_matrix[2,0] = gravity * np.cos(phi) * np.cos(theta)
    c_matrix[2,1] = -gravity * np.sin(phi) * np.sin(theta)
    c_matrix[2,2] = 0
    c_matrix[3,0] = -gravity * np.sin(phi) * np.cos(theta)
    c_matrix[3,1] = -gravity * np.sin(theta) * np.cos(phi)
    c_matrix[3,2] = 0

    return c_matrix


def get_next_state_saito(prior_state: Any, delta_t: float, angular_rotation: Any,
                        noise: Any, accelerometer: Any,
                        gravitational_constant: float = -9.81):
    phi, theta, psi = prior_state[:3, 0]

    omega_x, omega_y, omega_z = angular_rotation[:, 0]



    new_state = np.zeros((prior_state.shape[0], 1))

    new_state[0, 0] = phi + delta_t * omega_x + delta_t * (np.sin(phi) * np.tan(theta) * omega_y)  + np.cos(phi) * np.tan(theta) * omega_z * delta_t
    new_state[1, 0] = theta + delta_t * (np.cos(phi) * omega_y) - delta_t * (np.sin(psi) * omega_z)
    new_state[2, 0] = psi + delta_t * (np.sin(phi) * 1/np.cos(theta) * omega_y ) + delta_t * (np.cos(phi) * 1/np.cos(theta) * omega_z)
    return new_state


def get_measurement_saito(state_estimate: Any, accelerometer: Any, magnetometer: Any):
    phi, theta, psi = state_estimate[:, 0]
    a_x, a_y, a_z = accelerometer[:, 0]
    m_b_x, m_b_y, m_b_z = magnetometer[:, 0]

    rotation_matrix = np.zeros((3,3))
    rotation_matrix[0,0] = np.cos(theta)
    rotation_matrix[0,1] = np.sin(phi) * np.sin(theta)
    rotation_matrix[0,2] = np.cos(phi) * np.sin(theta)

    rotation_matrix[1,0] = 0
    rotation_matrix[1,1] = np.cos(phi)
    rotation_matrix[1,2] = -np.sin(phi)

    rotation_matrix[2,0] = -np.sin(theta)
    rotation_matrix[2,1] = np.sin(phi) * np.cos(theta)
    rotation_matrix[2,2] = np.cos(phi) * np.cos(theta)

    rotated_magnetometer = np.matmul(rotation_matrix, magnetometer)

    rotated_m_y = rotated_magnetometer[1, 0]
    rotated_m_x = rotated_magnetometer[0, 0]

    # had to flip signs here to get it to line up...where in paper it was neg m_y, pos m_x.
    psi = np.arctan2(rotated_m_y, -rotated_m_x)

    measurement = np.zeros((4,1))
    measurement[0,0] = psi
    measurement[1:4, 0] = accelerometer[0:3, 0]
    return measurement


def get_expected_measurement_saito(state_estimate: Any, gravity: float = 9.81):

    phi, theta, psi = state_estimate[:, 0]
    expected_measurement = np.zeros((4,1))
    expected_measurement[1, 0] = gravity * (-np.sin(theta))
    expected_measurement[2, 0] = gravity * np.sin(phi) * np.cos(theta)
    expected_measurement[3, 0] = gravity * np.cos(phi) * np.cos(theta)
    expected_measurement[0, 0] = psi

    return expected_measurement



def get_next_state(prior_state: Any, delta_t: float, angular_rotation: Any,
                        noise: Any, accelerometer: Any,
                        gravitational_constant: float = -9.81):
    #s_x, s_y, s_z = prior_state[0:3, 0]
    #v_x, v_y, v_z = prior_state[3:6, 0]
    phi, theta, psi = prior_state[:3, 0]

    omega_x, omega_y, omega_z = angular_rotation[:, 0]

    a_x, a_y, a_z = accelerometer[:, 0]

    new_state = np.zeros((prior_state.shape[0], 1))
    # new_state[0,0] = s_x + (v_x * math.cos(theta) * math.cos(psi)  + v_y * math.cos(theta) * math.sin(psi) - v_z * math.sin(theta)) * delta_t
    # new_state[1,0] = s_y + (v_x * (math.cos(phi) * math.sin(theta) * math.cos(psi) - math.cos(phi) * math.sin(psi)) + v_y * (math.sin(phi) * math.sin(theta) * math.sin(psi) + math.cos(phi) * math.cos(psi)) + v_z * (math.sin(phi) * math.cos(theta)) ) * delta_t
    # new_state[2,0] = s_z + (v_x * (math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)) + v_y * (math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)) + v_z * (math.cos(phi) * math.cos(theta)) ) * delta_t
    # new_state[3, 0] = v_x + delta_t * (a_x - gravitational_constant * math.sin(theta)  + omega_y * v_z - omega_z * v_y)
    # new_state[4, 0] = v_y + delta_t * (a_y + gravitational_constant * math.sin(phi) * math.cos(theta) - omega_x * v_z + omega_z * v_x)
    # new_state[5, 0] = v_z + delta_t * (a_z + gravitational_constant * math.cos(phi) * math.cos(theta) + omega_x * v_y - omega_y * v_x )
    new_state[0, 0] = phi + delta_t * (omega_x + np.sin(phi) * np.tan(theta) * omega_y + np.cos(phi) * np.tan(theta) * omega_z ) #+ math.sin(phi) * math.tan(omega_y) + math.cos(phi) * math.tan(theta) * omega_z
    new_state[1, 0] = theta + delta_t * (np.cos(phi) * omega_y - np.sin(phi) * omega_z) #math.cos(phi) * omega_y - math.sin(phi) * omega_z
    new_state[2, 0] = psi + delta_t * (np.sin(phi) * omega_y / np.cos(theta) + np.cos(phi) * omega_z / np.cos(theta) )#(math.sin(phi) / np.cos(theta)) * omega_y + (math.cos(phi) / np.cos(theta)) * omega_z

    return new_state + noise


def rotate_accelerometer_data(accel_data: Any):
    g_b_x, g_b_y, g_b_z = accel_data[:, 0]
    #accel_data[1, 0] = g_b_y
    return accel_data

def rotate_magnetometer_data(magn_data: Any):
    m_b_x, m_b_y, m_b_z = magn_data[:, 0]
    return magn_data


def calc_digital_compass(accelerometer: Any, magnitational_vector: Any):

    """
    phi, theta, psi = state_estimate[6:, 0]

    a, _, b = magnitational_vector[:, 0]

    digital_compass = np.zeros((3,1))
    digital_compass[0,0] = math.atan2((-gravity * math.sin(phi) * math.cos(theta)), (-gravity * math.cos(phi) * math.cos(theta)))
    digital_compass[1, 0] = math.atan2((gravity * math.sin(theta)),
                                       (-gravity * math.sin(phi) * math.cos(theta) * math.sin(phi) - gravity * math.cos(phi) * math.cos(theta) * math.cos(phi)))
    digital_compass[2, 0] = math.atan2((-math.sin(phi)), (a * math.cos(theta) * math.sin(theta) * math.cos(phi) * b))
    """
    sin_angle = 0
    cos_angle = 0
    m_b_x_derotated, m_b_y_derotated, m_b_z_derotated = 0, 0, 0
    m_b_x, m_b_y, m_b_z = rotate_magnetometer_data(magnitational_vector)

    g_b_x, g_b_y, g_b_z = rotate_accelerometer_data(accelerometer)

    phi = math.atan2(g_b_y, g_b_z)

    sin_angle = math.sin(phi)
    cos_angle = math.cos(phi)

    m_b_y_derotated = m_b_y * cos_angle - m_b_z * sin_angle
    m_b_z = m_b_y * sin_angle + m_b_z * cos_angle

    # theta = math.atan2(-g_b_x, (math.sqrt((g_b_y**2) + (g_b_z**2))))
    theta = math.atan2(-g_b_x, (g_b_y * sin_angle + g_b_z * cos_angle))
    sin_angle = math.sin(theta)
    cos_angle = math.cos(theta)

    m_b_x_derotated = m_b_x * cos_angle + m_b_z * sin_angle
    m_b_z_derotated = -m_b_x * sin_angle + m_b_z * cos_angle

    # m_hor_x = m_b_x * math.cos(theta) + m_b_y * math.sin(theta) * math.sin(phi) + m_b_z*math.sin(theta)*math.cos(phi)
    # m_hor_y = m_b_y * math.cos(phi) - m_b_z * math.sin(phi)
    #


    # h_hor = y_rotation @ x_rotation @ magnitational_vector
    # h_hor_y_2 = h_hor[1, 0]
    # h_hor_x_2 = h_hor[0, 0]

    # todo: why does the y here need to be pos and the x need to be negative?
    # todo: most formulas take opposite convention, are our axis rotated for magnetometer?
    psi = math.atan2(m_b_y_derotated, -m_b_x_derotated)

    orientation = np.zeros((3,1))
    orientation[0,0] = phi
    orientation[1,0] = theta
    orientation[2,0] = psi

    return orientation

def get_expected_measurement(state_estimate: Any, gravity: float = 9.81):
    phi, theta, psi = state_estimate[:, 0]
    expected_measurement = np.zeros(state_estimate.shape)
    expected_measurement[0, 0] = -gravity * (-np.sin(theta))
    expected_measurement[1, 0] = -gravity * np.sin(phi) * np.cos(theta)
    expected_measurement[2, 0] = -gravity * np.cos(phi) * np.cos(theta)
    return expected_measurement



def get_mu_sigma_extended_kalman(
        initial_state: Any, initial_covariance: Any, Q: Any,
        R: Any, process_noise_series: Any, gyroscope_data: Any, accelerometer_data: Any, time_step_data: Any, gravity: float,
        magnetometer_data: Any):

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
        magneto_measurement = magnetometer_data[i-1, :].reshape((3,1))
        delta_t = time_step_data[i - 1, 0] - prior_time_seconds
        prior_time_seconds = time_step_data[i - 1, 0]
        process_noise = process_noise_series[ i -1, :, :]

        mu_t_given_t_minus_one = get_next_state(prior_estimate, delta_t, gyro_measurement,
            process_noise, accel_measurement,
            gravity)

        a_matrix = get_a_matrix(prior_estimate, delta_t, gyro_measurement,
            gravity)

        sigma_t_given_t_minus_one = np.matmul(a_matrix, np.matmul(
            sigma_t_minus_one_given_t_minus_one, a_matrix.transpose())) + Q

        #measurement = calc_digital_compass(accel_measurement, magneto_measurement)
        measurement = get_expected_measurement(mu_t_given_t_minus_one, gravity)

        # error_vs_estimate = measurement - mu_t_given_t_minus_one[:3, :]
        error_vs_estimate = accel_measurement - measurement

        c_matrix = get_c_matrix(mu_t_given_t_minus_one, magneto_measurement)

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


def get_mu_sigma_extended_kalman_saito(
        initial_state: Any, initial_covariance: Any, Q: Any,
        R: Any, process_noise_series: Any, gyroscope_data: Any, accelerometer_data: Any, time_step_data: Any, gravity: float,
        magnetometer_data: Any):

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
        magneto_measurement = magnetometer_data[i-1, :].reshape((3,1))
        delta_t = time_step_data[i - 1, 0] - prior_time_seconds
        prior_time_seconds = time_step_data[i - 1, 0]
        process_noise = process_noise_series[ i -1, :, :]

        mu_t_given_t_minus_one = get_next_state_saito(prior_estimate, delta_t, gyro_measurement,
            process_noise, accel_measurement,
            gravity)

        a_matrix = get_a_matrix_saito(prior_estimate, delta_t, gyro_measurement,
            gravity)

        sigma_t_given_t_minus_one = np.matmul(a_matrix, np.matmul(
            sigma_t_minus_one_given_t_minus_one, a_matrix.transpose())) + Q

        #measurement = calc_digital_compass(accel_measurement, magneto_measurement)
        expected_measurement = get_expected_measurement_saito(mu_t_given_t_minus_one, gravity)

        measurement = get_measurement_saito(mu_t_given_t_minus_one, accel_measurement, magneto_measurement)

        # error_vs_estimate = measurement - mu_t_given_t_minus_one[:3, :]
        error_vs_estimate = measurement - expected_measurement

        c_matrix = get_c_matrix_saito(mu_t_given_t_minus_one, magneto_measurement)

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


def get_digital_compass_time_series(ground_truth: Any):
    orientation_measurements_digital_compass = np.zeros((ground_truth.shape[0], 3, 1))

    orientation_gyro = np.zeros((3, 1))
    prior_state = np.zeros((9, 1))
    for i in range(ground_truth.shape[0]):
        accel_measurement = ground_truth[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].values[i, :].reshape((3, 1))
        gyro_measurement = ground_truth[["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]].values[i, :].reshape((3, 1))
        magneto_measurement = ground_truth[["imu_magn_x", "imu_magn_y", "imu_magn_z"]].values[i, :].reshape((3, 1))
        orientation = calc_digital_compass(accel_measurement, magneto_measurement)

        orientation_measurements_digital_compass[i, :, :] = orientation
    return orientation_measurements_digital_compass


def get_estimate_plots_vs_ground_truth(ground_truth: Any, orientation_measurements_digital_compass: Any,
                                       orientation_estimate_ekf: Any):
    fig, ax = plt.subplots(3, 1, sharex=True)
    labels = ["x", "y", "z"]
    cols = ["orientation_x", "orientation_y", "orientation_z"]
    for i in range(3):
        axes_label = labels[i]
        col = cols[i]
        ax[i].plot(ground_truth["time_s"], ground_truth[col], label="true_orientation_{}_radians".format(axes_label))
        ax[i].plot(ground_truth["time_s"], orientation_measurements_digital_compass[:, i, 0],
                   label="compass_orientation_{}_radians".format(axes_label))
        ax[i].plot(ground_truth["time_s"], orientation_estimate_ekf[1:, i, 0],
                   label="ekf_{}_radians".format(axes_label))
        ax[i].legend(loc="lower right")

    # ax[0].set_title("Ground Truth vs Sensor Model {}".format(os.path.basename(DATASET_PATH)))

    fig.suptitle("Ground Truth vs Sensor Model, {} from imu_repo".format(os.path.basename(DATASET_PATH)))
    # ax[2].plot(ground_truth["time_s"].values[:estimates_to_plot], mu_estimates[1:estimates_to_plot+1, 8, 0], label="estimate_orientation_z_radians")
    return fig

def get_measurement_time_series(ground_truth: Any):
    fig, ax = plt.subplots(3, 2, sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        label = labels[i]
        ax[i][0].plot(ground_truth["time_s"], ground_truth["imu_acc_{}".format(label)],
                      label="accelerometer_{}".format(label))
        ax[i][0].set_ylabel("acceleration_{}".format(label))

        ax[i][1].plot(ground_truth["time_s"], ground_truth["imu_magn_{}".format(label)],
                      label="magnetometer_{}".format(label))
        ax[i][1].set_ylabel("magnetometer_{}".format(label))
    ax[0][0].set_title("Raw Accelerometer Data")
    ax[0][1].set_title("Raw Magnetometer Data")

    return fig

def main(orientation_estimates_path_name: str = None, measurement_estimates_path_name: str = None):
    ground_truth = load_dataset(DATASET_PATH)

    process_noise = get_gaussian_noise(
        ground_truth.shape[0], np.zeros((INITIAL_ESTIMATE.shape[0], 1)),
        Q)

    # mu_estimates, sigma_estimates, run_times = get_mu_sigma_extended_kalman(INITIAL_ESTIMATE, INITIAL_COVARIANCE,
    #                                     Q, R, process_noise, ground_truth[["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]].values[:, :], ground_truth[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].values[:, :], ground_truth[["time_s"]].values[:, :], -9.81,
    #    ground_truth[["imu_magn_x",
    #               "imu_magn_y", "imu_magn_z"]].values[:, :])

    mu_estimates, sigma_estimates, run_times = get_mu_sigma_extended_kalman_saito(INITIAL_ESTIMATE, INITIAL_COVARIANCE,
                                        Q, R, process_noise, ground_truth[["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]].values[:, :], ground_truth[["imu_acc_x", "imu_acc_y", "imu_acc_z"]].values[:, :], ground_truth[["time_s"]].values[:, :], -9.81,
       ground_truth[["imu_magn_x",
                  "imu_magn_y", "imu_magn_z"]].values[:, :])

    orientation_measurements_digital_compass = get_digital_compass_time_series(ground_truth)

    fig = get_estimate_plots_vs_ground_truth(ground_truth, orientation_measurements_digital_compass, mu_estimates)
    if orientation_estimates_path_name is None:
        plt.show()
    else:
        fig.savefig(orientation_estimates_path_name)

    fig = get_measurement_time_series(ground_truth)
    if measurement_estimates_path_name is None:
        plt.show()
    else:
        fig.savefig(measurement_estimates_path_name)

if __name__ == "__main__":
    main()
