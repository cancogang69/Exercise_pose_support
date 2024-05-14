import os
import time
import pandas as pd
import cv2
import numpy as np
import streamlit as st

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

from lib.models.pose_hrnet import get_pose_net
from lib.config import cfg
from lib.core.function import get_final_preds
from lib.utils.transforms import get_affine_transform
from lib.constants import demo_constant

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEBUG = True
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

NUM_KPTS = len(demo_constant.MPII_KEYPOINT_INDEXES)
SKELETON = demo_constant.SKELETON
SQUAT_PART = demo_constant.SQUAT_PART
SQUAT_KEYPART = demo_constant.SQUAT_KEYPART
STAGE_ANGLE = demo_constant.STAGE_ANGLE
COLOR = demo_constant.COLOR

# function definition
def cosine_similarity(a, b):
  return ( np.dot(a, b) / 
          (np.linalg.norm(a) * np.linalg.norm(b)) )

def angle_degree(a, b):
  return (np.arccos(cosine_similarity(a, b)) / 
          np.pi) * 180

def cosine_similarity(a, b):
  cos = ( np.dot(a, b) / 
          (np.linalg.norm(a) * np.linalg.norm(b)) )
  if cos < -1:
    return -1
  elif cos > 1:
    return 1
  else:
    return cos


def angle_degree(a, b):
  angle = (np.arccos(cosine_similarity(a, b)) / 
          np.pi) * 180
  if angle > 180:
    return 180.
  if angle < 0:
    return 0
  else:
    return angle

# @parameter
# part: contain 2 joint coordinate [[x0, y0], [x1, y1]]
def return_vector(part_id, skeleton):
  point_a = skeleton[part_id[0]]
  point_b = skeleton[part_id[1]]
  return point_a - point_b

def is_in_range(num, range1, range2):
  ceil = max(range1, range2)
  floor = min(range1, range2)
  return (num >= floor and num <= ceil)

def find_pose_id(ske_pred, keyparts, stage):
  assert ske_pred.shape == (NUM_KPTS, 2)
  sum = 0
  for keypart in keyparts:
    part_a, part_b = keypart
    vec_a = return_vector(SKELETON[part_a], ske_pred)
    vec_b = return_vector(SKELETON[part_b], ske_pred)

    pred_angle = angle_degree(vec_a, vec_b)
    sum += pred_angle

  sum /= len(keyparts)
  for i in range(len(stage)-1):
    if is_in_range(sum, stage[i], stage[i+1]):
      return i

def compare_skeleton(ske_true, ske_pred, excer_part, threshold= 10):
  assert ske_true.shape == (NUM_KPTS, 2)
  assert ske_pred.shape == (NUM_KPTS, 2)
  part_flag = []
  for part in excer_part:
    kpt_a, kpt_b = SKELETON[part]
    true_vec = ske_true[kpt_a] - ske_true[kpt_b]
    pred_vec = ske_pred[kpt_a] - ske_pred[kpt_b]
    angle = angle_degree(true_vec, pred_vec)
    part_flag.append(True if angle <= threshold else False)
  return part_flag

def draw_pose(keypoints, part_flag, excer_part, img, joint_thickness=6):
  assert keypoints.shape == (NUM_KPTS, 2)
  for i, part in enumerate(excer_part):
    kpt_a, kpt_b = SKELETON[part]
    c = COLOR["blue"] if part_flag[i] else COLOR["red"]
    x_a, y_a = keypoints[kpt_a]
    x_b, y_b = keypoints[kpt_b]
    cv2.circle(img, (int(x_a), int(y_a)), joint_thickness, COLOR["green"], -1)
    cv2.circle(img, (int(x_b), int(y_b)), joint_thickness, COLOR["green"], -1)
    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), c, 2)

def draw_bbox(box, img):
    """
    draw the detected bounding box on the image.
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
  results = model(img)
  
  result_df = results.pandas().xyxy[0]
  person_df = result_df[result_df["name"] == "person"]
  person_df = person_df[person_df["confidence"] > threshold]
  person_bboxes = person_df.iloc[:, :4].values

  return person_bboxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
  rotation = 0

  # pose estimation transformation
  trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
  model_input = cv2.warpAffine(
    image,
    trans,
    (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
    flags=cv2.INTER_LINEAR)
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
  ])

  # pose estimation inference
  model_input = transform(model_input).unsqueeze(0)
  # switch to evaluate mode
  pose_model.eval()
  with torch.no_grad():
    # compute output heatmap
    output = pose_model(model_input)
    preds, _ = get_final_preds(
      cfg,
      output.clone().cpu().numpy(),
      np.asarray([center]),
      np.asarray([scale]))

    return preds


def box_to_center_scale(box, model_image_width, model_image_height):
  """convert a box to center,scale information required for pose transformation
  Parameters
  ----------
  box : list of tuple
      list of length 2 with two tuples of floats representing
      bottom left and top right corner of a box
  model_image_width : int
  model_image_height : int

  Returns
  -------
  (numpy array, numpy array)
      Two numpy arrays, coordinates for the center of the box and the scale of the box
  """
  center = np.zeros((2), dtype=np.float32)

  x_min, y_min, x_max, y_max = box
  box_width = x_max - x_min
  box_height = y_max - y_min
  center[0] = x_min + box_width * 0.5
  center[1] = y_min + box_height * 0.5

  aspect_ratio = model_image_width * 1.0 / model_image_height
  pixel_std = 200

  if box_width > aspect_ratio * box_height:
    box_height = box_width * 1.0 / aspect_ratio
  elif box_width < aspect_ratio * box_height:
    box_width = box_height * aspect_ratio
  scale = np.array(
    [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
    dtype=np.float32)
  if center[0] != -1:
    scale = scale * 1.25

  return center, scale

def update_config(default_cfg, config_file):
  default_cfg.defrost()
  default_cfg.merge_from_file(config_file)
  default_cfg.freeze()

# streamlit cache    
@st.cache_resource
def load_yolo(version):
  return torch.hub.load('ultralytics/yolov5', version, _verbose=False)

@st.cache_resource
def load_hrnet(config_path):
  update_config(cfg, config_path)

  pose_model = get_pose_net(cfg, is_train=False)

  if cfg.TEST.MODEL_FILE:
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
  else:
    print('expected model defined in config at TEST.MODEL_FILE')

  pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
  pose_model.to(CTX)
  pose_model.eval()

  return pose_model

@st.cache_resource
def load_true_poses(path):
  joints_path = os.path.join(path, "joints.csv")
  if not os.path.exists(joints_path):
    st.write("Can't find joint file!")
    return None, None
  
  img_path = os.path.join(path, "images")
  if not os.path.exists(joints_path):
    st.write("Can't find image files!")
    return None, None
  
  img_paths = os.listdir(img_path)
  imgs = [cv2.imread(os.path.join(img_path, p)) for p in img_paths]
  skes = pd.read_csv(joints_path).to_numpy().reshape(len(imgs), NUM_KPTS, 2)
  return imgs, skes

def stop_processing():
  st.session_state["start"] = False
  if video_cap:
    video_cap.release()
    video_window.image([])

def start_processing():
  st.session_state["start"] = True

# model loading and state init
if "start" not in st.session_state:
  st.session_state["start"] = False

box_model = load_yolo("yolov5s")
pose_model = load_hrnet("./demo/w32_256x256_mpii.yaml")
true_pose_images, true_pose_skes = load_true_poses("./squat")


# Sidebar
model_selectbox = st.sidebar.selectbox("Model",
  ["HRNet w32"]
)
box_model_threshold = st.sidebar.number_input("Detetor threshold", min_value= 0.1, max_value=1., value=0.7)
angle_threshold = st.sidebar.number_input("Angle threshold", min_value= 0, max_value=180, value= 5)
true_pose_threshold = st.sidebar.number_input("True pose threshold", min_value= 0, max_value=100, value= 75)
video_data = st.sidebar.file_uploader("Choose video", type=["mp4"])
temp_video = "./temp.mp4"
video_cap = None


# main page
col1, col2 = st.columns(2)
col1.title("Your pose")
video_window = col1.image([])

col2.title("True pose")
true_pose_window = col2.image([])


if video_data and len(true_pose_images) and len(true_pose_skes):
  count = 0
  st.sidebar.button("Start", on_click= start_processing)

  if st.session_state["start"]:
    st.sidebar.button("Cancel Execution", on_click= stop_processing)
    
    with open(temp_video, "wb") as outfile:
      outfile.write(video_data.getbuffer())
    video_cap = cv2.VideoCapture(temp_video)
    while True:
      pose_id = 0
      rep = count // len(true_pose_images)
      
      ret, image_bgr = video_cap.read()
      if ret:
        # true_pose_window.image(cv2.cvtColor(true_pose_images[pose_id], cv2.COLOR_BGR2RGB))
        last_time = time.time()
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred_boxes = get_person_detection_boxes(box_model, image, threshold= box_model_threshold)

        for box in pred_boxes:
          center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
          image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
          pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
          for skeleton in pose_preds:
            pose_id = find_pose_id(skeleton, SQUAT_KEYPART, STAGE_ANGLE)
            true_skeleton = true_pose_skes[pose_id]
            print(pose_id)
            part_flag = compare_skeleton(true_skeleton, skeleton, SQUAT_PART, threshold= angle_threshold)
            draw_pose(skeleton, part_flag, SQUAT_PART, image_bgr)
            true_percent = sum(part_flag) / len(part_flag)
            if true_percent >= true_pose_threshold:
              count += 1

        if DEBUG:
          fps = 1/(time.time() - last_time)
          image = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR["blue"], 2)

        true_pose_window.image(cv2.cvtColor(true_pose_images[pose_id], cv2.COLOR_BGR2RGB))
        result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_window.image(result_image)
      else:
        break
      
    stop_processing()