MPII_KEYPOINT_INDEXES = {
  0: "right ankle",
  1: "right knee",
  2: "right hip", 
  3: "left hip", 
  4: "left knee", 
  5: "left ankle",
  6: "pelvis", 
  7: "thorax", 
  8: "upper neck", 
  9: "head top", 
  10: "right wrist",
  11: "right elbow", 
  12: "right shoulder", 
  13: "left shoulder", 
  14: "left elbow",
  15: "left wrist"
}

SKELETON = {
  "left_lower_leg": [0, 1], 
  "left_thigh": [2, 1], 
  "left_hip": [2, 6], 
  "right_lower_leg": [5, 4],
  "right_thigh": [3, 4],
  "right_hip": [3, 6],
  "torso": [6, 7], 
  "neck": [7, 8],             #actually it's thorax - upper neck 
  "head": [8, 9],
  "right_forearm": [10, 11],
  "right_upper_arm": [12, 11], 
  "right_shoulder": [12, 7],
  "left_forearm": [15, 14],
  "left_upper_arm": [13, 14], 
  "left_shoulder": [13, 7]
}

SQUAT_PART = ["left_lower_leg", "left_thigh", "right_lower_leg",
              "right_thigh", "torso"]

SQUAT_KEYPART = [
  ["left_lower_leg", "left_thigh"],
  ["right_lower_leg", "right_thigh"]
]

JUMPING_JACK_PART = ["left_lower_leg", "left_thigh", 
  "right_lower_leg", "right_thigh",
  "torso", "right_forearm", "right_upper_arm",
  "left_forearm", "left_upper_arm"]

JUMPING_JACK_KEYPART = [
  ["left_upper_arm", "torso"],
  ["right_upper_arm", "torso"]
]

SQUAT_STAGE_ANGLE = [[180, 170], [170, 155],
               [155, 137], [137, 114], [114, 0]]

JUMPING_STAGE_ANGLE = [[0, 10], [10, 35], [35, 64],
                       [64, 104], [104, 180]]

COLOR = {
  "red": [0, 0, 255],
  "blue": [255, 0, 0],
  "green": [0, 255, 0]
}