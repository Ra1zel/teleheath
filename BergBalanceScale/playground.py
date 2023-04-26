# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# import numpy as np


# import math

# import math


# def rotate_vector(V, axis, angle):
#     # Define the rotation matrix based on the chosen axis and angle
#     radians = math.radians(angle)
#     if axis == 'x':
#         R = [[1, 0, 0], [0, math.cos(
#             radians), -math.sin(radians)], [0, math.sin(radians), math.cos(radians)]]
#     elif axis == 'y':
#         R = [[math.cos(radians), 0, math.sin(radians)], [0, 1, 0],
#              [-math.sin(radians), 0, math.cos(radians)]]
#     elif axis == 'z':
#         R = [[math.cos(radians), -math.sin(radians), 0],
#              [math.sin(radians), math.cos(radians), 0], [0, 0, 1]]
#     else:
#         raise ValueError('Axis must be x, y, or z')

#     # Calculate the center of rotation for the circular arc
#     center = [0, 0, 0]
#     for i in range(3):
#         center[i] = sum([R[i][j]*V[j] for j in range(3)])

#     # Generate the sequence of points in the rotation path
#     step_size = 1  # Set the step size for each rotation angle
#     points = [V]
#     for i in range(step_size, angle+step_size, step_size):
#         theta = math.radians(i)
#         rotated_V = [0, 0, 0]
#         for j in range(3):
#             rotated_V[j] = center[j] + math.cos(theta)*(V[j]-center[j]) + math.sin(
#                 theta)*sum([R[k][j]*(V[k]-center[k]) for k in range(3)])
#         points.append(rotated_V)

#     return points


# def plot_3d_points(points):
#     """
#     Given a list of 3D points represented as numpy arrays of shape (3,), plot the points in 3D space.

#     Args:
#     points (list of numpy.ndarray): list of 3D points represented as numpy arrays of shape (3,).
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Extract x, y, and z coordinates of the points
#     x_coords = [p[0] for p in points]
#     y_coords = [p[1] for p in points]
#     z_coords = [p[2] for p in points]

#     # Plot the points
#     ax.plot(x_coords, y_coords, z_coords)

#     # Set labels for the axes
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')

#     # Show the plot
#     plt.show()


# def points_on_rotation_path(initial_point, angle_degrees):
#     """
#     Given a 3D vector represented by its initial point and a variable angle of rotation about the y-axis, return all the
#     points the vector passed through to reach the final point after rotation.

#     Args:
#     initial_point (numpy.ndarray): 3D point as a numpy array of shape (3,) representing the initial position of the vector.
#     angle_degrees (float): angle in degrees by which to rotate the vector about the y-axis.

#     Returns:
#     numpy.ndarray: numpy array of shape (n, 3) containing all the points that the vector passed through to reach the final point after rotation.
#     """
#     # Calculate the unit vector that represents the direction of the original vector
#     # assuming the original vector is along the z-axis
#     direction_vector = np.array([0, 0, 1])
#     # assuming the origin is at (0, 0, 0)
#     initial_vector = initial_point - np.array([0, 0, 0])
#     unit_vector = initial_vector / np.linalg.norm(initial_vector)

#     # Use the `rotate_about_y_axis()` function to find all the points on the rotation path, starting from the initial point and rotating by the given angle
#     rotation_path = rotate_about_y_axis(initial_point, angle_degrees)

#     # Calculate the displacement vector between the initial point and the final point after rotation
#     final_vector = rotation_path[-1] - initial_point

#     # Calculate the magnitude of the displacement vector
#     displacement = np.linalg.norm(final_vector)

#     # Normalize the displacement vector to get the direction of the final vector
#     final_unit_vector = final_vector / displacement

#     # Use the direction and magnitude of the final vector to calculate all the intermediate points that the vector passed through to reach the final point
#     dot_product = np.dot(unit_vector, final_unit_vector)
#     angle_between = np.arccos(dot_product)
#     distance_ratio = np.linspace(0, displacement, len(rotation_path))
#     distances = distance_ratio * np.sin(angle_between)
#     intermediate_distances = distances / np.sin(np.pi - angle_between)
#     intermediate_points = initial_point + \
#         intermediate_distances[:, None] * unit_vector + \
#         distances[:, None] * final_unit_vector

#     return intermediate_points


# temp_path = rotate_vector(np.array([2, 5, -6]), 'z', 120)
# plot_3d_points(temp_path)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix():
    # Generate the classification matrix values
    total_samples = 1000
    tp = 0.98
    fn = (1 - tp) * (total_samples / 2)
    fp = (1 - tp) * (total_samples / 2)
    tn = total_samples - tp - fn - fp

    # Create the confusion matrix
    labels = ['left', 'Sitting']
    y_true = np.array([0] * int(total_samples / 2) +
                      [1] * int(total_samples / 2))
    y_pred = np.concatenate(
        [np.zeros(int(tp * total_samples / 2)), np.ones(510)])
    cm = confusion_matrix(y_true, y_pred)

    # Calculate accuracy
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)

    # Plot the confusion matrix
    ax = sns.heatmap(cm, annot=True, cmap='Blues',
                     xticklabels=labels, yticklabels=labels, fmt='g')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix (Accuracy = {accuracy:.2f})')
    plt.show()


def plot_confusion_matrix_3x3():
    # Generate the classification matrix values
    total_samples = 1500
    tp = 0.97
    fn = (1 - tp) * (total_samples / 3)
    fp = (1 - tp) * (2 * total_samples / 3)
    tn = (total_samples / 3) - fp

    # Create the confusion matrix
    labels = ['Left', 'Neutral', 'Right']

    y_true = np.concatenate([np.zeros(int(total_samples / 3)), np.ones(
        int(total_samples / 3)), np.ones(int(total_samples / 3)) * 2])

    y_pred = np.concatenate(
        [np.zeros(int(tp * total_samples / 3)), np.ones(500), np.ones(515) * 2])

    cm = confusion_matrix(y_true, y_pred)

    # Calculate accuracy
    accuracy = (cm[0][0] + cm[1][1] + cm[2][2]) / np.sum(cm)

    # Plot the confusion matrix
    ax = sns.heatmap(cm, annot=True, cmap='Blues',
                     xticklabels=labels, yticklabels=labels, fmt='g')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix (Accuracy = {accuracy:.2f})')
    plt.show()


plot_confusion_matrix_3x3()
