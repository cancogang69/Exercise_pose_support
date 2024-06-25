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
  "right_upper_arm": [11, 12], 
  "right_shoulder": [12, 7],
  "left_forearm": [15, 14],
  "left_upper_arm": [14, 13], 
  "left_shoulder": [13, 7]
}

SQUAT_PART = ["left_lower_leg", "left_thigh", "right_lower_leg",
              "right_thigh", "torso"]

SQUAT_KEYPART = [
  ["left_lower_leg", "left_thigh"],
  ["right_lower_leg", "right_thigh"]
]

s_k = [ [[0, 1], [2, 1], 
         [[5, 4], [3, 4]]] ]

STAGE_ANGLE = [[180,170], [170, 155],
               [155, 137], [137, 114],
               [114, 0], [0, 114],
               [114, 148], [148, 168],
               [168, 180]]

COLOR = {
  "red": [0, 0, 255],
  "blue": [255, 0, 0],
  "green": [0, 255, 0]
}