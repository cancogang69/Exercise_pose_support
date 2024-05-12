SKELETON = {
    "left_lower_leg": [0, 1], 
    "left_thigh": [1, 2], 
    "left_hip": [2, 6], 
    "right_lower_leg": [5, 4],
    "right_thigh": [4, 3],
    "right_hip": [3, 6],
    "torso": [6, 7], 
    "neck": [7, 8],             #actually it's thorax - upper neck 
    "head": [8, 9],
    "right_forearm": [10, 11],
    "right_upper_arm": [11, 12], 
    "right_shoulder": [12, 7],
    "left_forearm": [15, 14],
    "left_upper_arm": [14, 13], 
    "left_shoulder": [13, 7]
}

SQUAT_PART = ["left_lower_leg", "left_thigh", "right_lower_leg",
              "right_thigh", "torso"]

def cosine_similarity(a, b):
  return ( np.dot(a, b) / 
          (np.linalg.norm(a) * np.linalg.norm(b)) )

def angle_degree(a, b):
  return (np.arccos(cosine_similarity(a, b)) / 
          np.pi) * 180

def draw_pose(keypoints, part_flag, excer_part, img, joint_thickness=6):
  assert keypoints.shape == (NUM_KPTS, 2)
  for i, part in enumerate(excer_part):
    kpt_a, kpt_b = SKELETON[part]
    color = blue_value if part_flag[i] else red_value
    x_a, y_a = keypoints[kpt_a]
    x_b, y_b = keypoints[kpt_b]
    cv2.circle(img, (int(x_a), int(y_a)), joint_thickness, red_value, -1)
    cv2.circle(img, (int(x_b), int(y_b)), joint_thickness, red_value, -1)
    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)

def compare_skeleton(ske_true, ske_pred, excer_part, threshold=10):
  assert ske_pred.shape == (NUM_KPTS, 2)
  assert ske_true.shape == (NUM_KPTS, 2)
  part_flag = []
  for part in excer_part:
    kpt_a, kpt_b = SKELETON[part]
    true_vec = ske_true[kpt_a] - ske_true[kpt_b]
    pred_vec = ske_pred[kpt_a] - ske_pred[kpt_b]
    angle = angle_degree(true_vec, pred_vec)
    part_flag.append(True if angle <= threshold else False)
  return part_flag