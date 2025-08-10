import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import math

pyautogui.FAILSAFE = False
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.75, min_tracking_confidence=0.75)
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

calib_points = [
    (int(screen_w * x), int(screen_h * y)) for y in [0.1, 0.5, 0.9] for x in [0.1, 0.5, 0.9]
]

def get_iris_center(landmarks, w, h, left=True):
    ids = [474, 475, 476, 477] if left else [469, 470, 471, 472]
    points = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in ids]
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

def get_eye_landmarks(landmarks, w, h, left=True):
    if left:
        idxs = [33, 133, 159, 145, 27, 130]
    else:
        idxs = [362, 263, 386, 374, 257, 359]
    return [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in idxs]

def get_head_pose(landmarks, w, h):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    image_points = np.array([
        (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),
        (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h),
        (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h),
        (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),
        (landmarks.landmark[287].x * w, landmarks.landmark[287].y * h),
        (landmarks.landmark[57].x * w, landmarks.landmark[57].y * h),
    ], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points,
                                                                camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return (pitch, yaw, roll)

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def median_filter(data, k=5):
    if len(data) < k:
        return data[-1]
    else:
        return np.median(data[-k:], axis=0)

calibration_data_iris = []
calibration_data_screen = []

cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for idx, point in enumerate(calib_points):
    samples = []
    start = time.time()
    while time.time() - start < 4:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(blank, point, 40, (0, 0, 255), -1)
        text = f"Focus on the red dot {idx + 1}/{len(calib_points)}"
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        cv2.putText(blank, text, ((screen_w - ts[0]) // 2, screen_h // 2 - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow("Calibration", blank)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            left_iris = get_iris_center(landmarks, w, h, True)
            right_iris = get_iris_center(landmarks, w, h, False)
            avg_iris = ((left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2)
            samples.append(avg_iris)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    if samples:
        calibration_data_iris.extend(samples)
        calibration_data_screen.extend([point] * len(samples))
    else:
        calibration_data_iris.append((0, 0))
        calibration_data_screen.append(point)

cv2.destroyWindow("Calibration")

X = np.array(calibration_data_iris)
y = np.array(calibration_data_screen)

model_x = LinearRegression()
model_y = LinearRegression()
model_x.fit(X, y[:, 0])
model_y.fit(X, y[:, 1])

smooth_factor_fast = 0.5
smooth_factor_slow = 0.15
prev_smoothed = None
prev_mouse_pos = pyautogui.position()
movement_threshold = 7
iris_history = []

cv2.namedWindow("Eye Mouse Control")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_iris = get_iris_center(landmarks, w, h, True)
        right_iris = get_iris_center(landmarks, w, h, False)
        avg_iris = ((left_iris[0] + right_iris[0]) / 2, (left_iris[1] + right_iris[1]) / 2)

        iris_history.append(avg_iris)
        if len(iris_history) > 7:
            iris_history.pop(0)

        filtered_iris = median_filter(iris_history, k=5)

        if prev_smoothed is None:
            smoothed = filtered_iris
        else:
            speed = np.linalg.norm(np.subtract(filtered_iris, prev_smoothed))
            factor = smooth_factor_fast if speed > 3 else smooth_factor_slow
            smoothed = (prev_smoothed[0] * (1 - factor) + filtered_iris[0] * factor,
                        prev_smoothed[1] * (1 - factor) + filtered_iris[1] * factor)

        prev_smoothed = smoothed

        pred_x = int(model_x.predict([[smoothed[0], smoothed[1]]])[0])
        pred_y = int(model_y.predict([[smoothed[0], smoothed[1]]])[0])

        pose = get_head_pose(landmarks, w, h)
        if pose is not None:
            pitch, yaw, roll = pose
            pred_x += int(yaw * 5)
            pred_y += int(pitch * 5)

        pred_x = clamp(pred_x, 0, screen_w - 1)
        pred_y = clamp(pred_y, 0, screen_h - 1)

        dist = np.hypot(pred_x - prev_mouse_pos[0], pred_y - prev_mouse_pos[1])

        if dist > movement_threshold:
            ease_factor = 0.2
            new_x = int(prev_mouse_pos[0] + (pred_x - prev_mouse_pos[0]) * ease_factor)
            new_y = int(prev_mouse_pos[1] + (pred_y - prev_mouse_pos[1]) * ease_factor)
            pyautogui.moveTo(new_x, new_y)
            prev_mouse_pos = (new_x, new_y)

        for x_, y_ in [left_iris, right_iris]:
            cv2.circle(frame, (int(x_), int(y_)), 5, (0, 255, 0), -1)

        if pose is not None:
            cv2.putText(frame, f"Pitch:{pitch:.1f} Yaw:{yaw:.1f} Roll:{roll:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    else:
        cv2.putText(frame, "Face or eyes not detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Eye Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
