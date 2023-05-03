import math
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_angles(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_all_angles(required_coordinates):
    results = []
    for coordinate in required_coordinates:
        results.append(calculate_angles(coordinate[0],coordinate[1],coordinate[2]))
    return results

def get_coordinates(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x , landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x , landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x , landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    right_shoulder =[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    right_wrist =[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x , landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]
    return [
        [left_shoulder,left_elbow,left_wrist],
        [right_shoulder,right_elbow,right_wrist],
        [left_hip,left_shoulder,left_elbow],
        [right_hip,right_shoulder,right_elbow],
        [left_shoulder,left_hip,left_knee],
        [right_shoulder,right_elbow,right_knee],
        [left_hip,left_knee,left_ankle],
        [right_hip,right_knee,right_ankle],
        [left_knee,left_ankle,left_foot_index],
        [right_knee,right_ankle,right_foot_index],
    ]

def rotate_points_around_y(p1, p2):
    # Convert the points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Calculate the angle between the two points and the y-axis
    angle = np.arctan2(p2[2] - p1[2], p2[0] - p1[0])

    # Calculate the rotation matrix around the y-axis
    R = np.array([[np.cos(angle), 0, -np.sin(angle)],
                  [0, 1, 0],
                  [np.sin(angle), 0, np.cos(angle)]])

    # Calculate the points along the path of the rotation
    num_steps = 360
    step_size = np.pi * 2 / num_steps
    points = []
    for i in range(num_steps + 1):
        # Calculate the rotation angle for this step
        rot_angle = i * step_size

        # Apply the rotation to the first point
        p1_rotated = R.dot(p1)

        # Calculate the rotation matrix for this step
        R_step = np.array([[np.cos(rot_angle), 0, np.sin(rot_angle)],
                           [0, 1, 0],
                           [-np.sin(rot_angle), 0, np.cos(rot_angle)]])

        # Apply the rotation matrix to both points
        p1_rotated = R_step.dot(p1_rotated)
        p2_rotated = R_step.dot(p2)

        # Add the rotated points to the list of points
        points.append((p1_rotated, p2_rotated))

    return np.array(points)

def separate_coordinates(coords):
    x1_coords = []
    y1_coords = []
    z1_coords = []
    x2_coords = []
    y2_coords = []
    z2_coords = []

    for coord in coords:
        x1_coords.append(coord[0][0])
        y1_coords.append(coord[0][1])
        z1_coords.append(coord[0][2])
        x2_coords.append(coord[1][0])
        y2_coords.append(coord[1][1])
        z2_coords.append(coord[1][2])

    return x1_coords, y1_coords, z1_coords, x2_coords, y2_coords, z2_coords

def calc_rotation_path_points(coordinate_1, coordinate_2):
    res = rotate_points_around_y(coordinate_1, coordinate_2)
    x1_coords, y1_coords, z1_coords, x2_coords, y2_coords, z2_coords = separate_coordinates(
        res)
    return x1_coords, y1_coords, z1_coords, x2_coords, y2_coords, z2_coords

def is_point_present(points_array, a, small_distance):
    points_array = np.array(points_array)
    a = np.array(a)
    dist = np.linalg.norm(points_array - a, axis=1)
    return np.any(dist <= small_distance)

def merge_coordinates(xs, ys, zs):
    return [[xs[i], ys[i], zs[i]] for i in range(len(xs))]

def angle_between_vectors(v1, v2):
    if np.array_equal(v1, v2):
        return 0

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(cosine_angle)
    # if dot_product < 0:
    #     angle *= -1
    cross_product = np.cross(v1, v2)
    if cross_product[2] < 0:
        angle = 2 * math.pi - angle
    return angle

def vector_from_b_to_a(a, b):
    # Calculate the vector from b to a
    vector = np.array(a) - np.array(b)
    return vector

def radians_to_degrees(angle_in_radians):
    angle_in_degrees = angle_in_radians * 180 / math.pi
    return angle_in_degrees

def calculate_rotation_angle(current_left_val, current_right_val, original_vector):
    curr_vector = vector_from_b_to_a(current_right_val, current_left_val)
    theta = angle_between_vectors(original_vector, curr_vector)
    return radians_to_degrees(theta)

def signed_angle(A, B):
    # Calculate the angle between A and B
    cos_angle = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    # Calculate the direction of the angle using the cross product of A and B
    cross_product = np.cross(A, B)
    direction = np.sign(cross_product[2])

    # Return the signed angle
    return angle * direction

def check_distance(p1, p2, threshold):
    """
    Checks the distance between two 3D points and returns True if the distance is less than or equal to the threshold,
    and False otherwise.
    :param p1: the first 3D point as a tuple (x, y, z)
    :param p2: the second 3D point as a tuple (x, y, z)
    :param threshold: the threshold value as a float
    :return: True if the distance between p1 and p2 is less than or equal to threshold, False otherwise
    """
    distance = math.sqrt((p1[0] - p2[0]) ** 2 +
                         (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return distance <= threshold

def angle_between_fixed_and_rotating_vector(fixed_vector, rotating_vector):
    # ref_vector = (1, 0, 0)  # Reference vector to determine rotation direction
    v1 = [a - b for a, b in zip(rotating_vector, fixed_vector)]
    v2 = [a - b for a, b in zip(ref_vector, fixed_vector)]
    dot_product = sum((a*b) for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a**2 for a in v1))
    magnitude_v2 = math.sqrt(sum(a**2 for a in v2))
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.degrees(math.acos(cos_angle))
    if dot_product < 0:
        angle = 360 - angle
    return angle

def play_video_from_file(filepath):
    # Create a VideoCapture object and read from the input file
    cap = cv2.VideoCapture(filepath)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is not read correctly, break
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Press Q on keyboard to exit
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def calc_rotation_direction(arr):
    n = len(arr)
    mid = n // 2
    first_half = arr[:mid]
    second_half = arr[mid:]
    ones_first_half = first_half.count(1)
    ones_second_half = second_half.count(1)
    if ones_first_half > (n - 1) // 2 - ones_second_half:
        return "anti-clockwise"
    elif ones_first_half < (n - 1) // 2 - ones_second_half:
        return "clockwise"
    else:
        return "no rotation needed"

def apply_mediapipe_holistic_model(filepath):
    cap = cv2.VideoCapture(filepath)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())

            try:
            ######################################################
            #Extract Joint coordinates
                my_landmarks = results.pose_landmarks.landmark
                joint_coordinates = get_coordinates(my_landmarks)
            #calculate joint angles
                req_angles = get_all_angles(joint_coordinates)
                req_angles = np.around(req_angles,2)
            #display joint angles
            ###########################
            #############################

            except Exception as e:
                print('error',e)
                pass

            # Flip the image horizontally for a selfie-view display.

            cv2.imshow('MediaPipe Holistic',image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()




def points_on_line(a, b, num_points=10):
    # Create a unit vector in the direction from a to b
    v = (b - a) / np.linalg.norm(b - a)

    # Create an array of t values from 0 to 1 with num_points elements
    t_values = np.linspace(0, 1, num_points)

    # Calculate the coordinates of the points on the line using the parameterization r = a + t*v
    points = np.array([a + t*v for t in t_values])

    return points


def distance(a, b):
    # Calculate the difference between the two coordinates
    diff = b - a

    # Calculate the Euclidean distance between the two coordinates using numpy's norm function
    dist = np.linalg.norm(diff)

    return dist
