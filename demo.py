from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil
import time
from datetime import datetime

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

from lib.models.pose_hrnet import get_pose_net
from lib.config import cfg
from lib.config import update_config
from lib.core.function import get_final_preds
from lib.utils.transforms import get_affine_transform

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

SKELETON = [
    [0, 1], [1, 2], [2, 6], 
    [5, 4], [4, 3], [3, 6],
    [6, 7], [7, 8], [8, 9],
    [10, 11], [11, 12], [12, 7],
    [15, 14], [14, 13], [13, 7]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = len(MPII_KEYPOINT_INDEXES.keys())

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints, img, joint_thickness=6):
    assert keypoints.shape == (NUM_KPTS, 2)
    for i, (kpt_a, kpt_b) in enumerate(SKELETON):
        x_a, y_a = keypoints[kpt_a]
        x_b, y_b = keypoints[kpt_b]
        cv2.circle(img, (int(x_a), int(y_a)), joint_thickness, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), joint_thickness, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

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
    person_bboxs = person_df.iloc[:, :4].values

    return person_bboxs


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

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='demo/w32_256x256_mpii.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam',action='store_true')
    parser.add_argument('--image',type=str)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--fps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_output_file_name(input_path, output_path):
    input_name, input_type = os.path.basename(input_path).split(".")
    output_name = f"{input_name}_pose.{input_type}"
    save_path = os.path.join(output_path, output_name)
    return save_path

def get_datetime():
    now = datetime.now()
    return str(now).replace(" ", "_")

def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    output_path = os.path.join(".", "infer_output")
    create_dir(output_path)

    output_types = ["images", "videos", "webcam"]
    output_type_paths = [os.path.join(output_path, media_type) for media_type in output_types]
    print(output_type_paths)
    for path in output_type_paths:
        create_dir(path)
    
    box_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', _verbose=False)
    box_model_threshold = 0.7

    pose_model = get_pose_net(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video or an image or webcam 
    if args.webcam:
        vidcap = cv2.VideoCapture(0)
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
    elif args.image:
        image_bgr = cv2.imread(args.image)
    else:
        print('please use --video or --webcam or --image to define the input.')
        return 

    if args.webcam or args.video:
        if args.write:
            if args.video:
                save_path = get_output_file_name(args.video, output_type_paths[1])
            else:
                now = get_datetime()
                save_path = get_output_file_name(now, output_type_paths[2])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, 24.0, (int(vidcap.get(3)), int(vidcap.get(4))))
        while True:
            ret, image_bgr = vidcap.read()
            if ret:
                last_time = time.time()
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # object detection box
                pred_boxes = get_person_detection_boxes(box_model, image, threshold= box_model_threshold)

                # pose estimation
                for box in pred_boxes:
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                    for kpt in pose_preds:
                        draw_pose(kpt, image_bgr) # draw the poses

                if args.fps:
                    fps = 1/(time.time()-last_time)
                    image = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                if args.write:
                    out.write(image_bgr)

                cv2.imshow('demo',image_bgr)
                if cv2.waitKey(1) & 0XFF==ord('q'):
                    break
            else:
                print('cannot load the video.')
                break

        cv2.destroyAllWindows()
        vidcap.release()
        if args.write:
            print('video has been saved as {}'.format(save_path))
            out.release()

    else:
        # estimate on the image
        last_time = time.time()
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        pred_boxes = get_person_detection_boxes(box_model, image, threshold= box_model_threshold)

        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
            for kpt in pose_preds:
                draw_pose(kpt, image_bgr) # draw the poses
        
        if args.fps:
            fps = 1/(time.time()-last_time)
            image = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        if args.write:
            save_path = get_output_file_name(args.image, output_type_paths[0])
            cv2.imwrite(save_path, image_bgr)
            print('the result image has been saved as {}'.format(save_path))

        cv2.imshow('demo', image_bgr)
        if cv2.waitKey(0) & 0XFF==ord('q'):
            cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()
