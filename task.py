import numpy as np

import torch
import os
import argparse
import pathlib
import warnings

import cv2

import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

def get_device(device):
    if device == 'cpu':
        return torch.device('cpu')
    elif device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='path to input video',
        default='inference/input/crowd.mp4'
    )
    parser.add_argument(
        "--device", default="", help="cuda or cpu")

    parser.add_argument(
        "--thresh", type=float, default=0.8, help="confidence threshold")
    parser.add_argument("--imgsize", type = int, default = 512, help = "image size (pixels)")
    args = vars(parser.parse_args())
    return args

def main(args):
    DEVICE = get_device(args['device'])
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights)
    model.to(DEVICE).eval()
    detection_threshold = args['thresh']
    RESIZE_TO = (args['imgsize'], args['imgsize'])

    cap = cv2.VideoCapture(args['input'])

    if (cap.isOpened() == False):
      warnings.warn('Error while trying to read video. Please check path again')


    save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
    out = cv2.VideoWriter(f"inference/output/{save_name}.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          RESIZE_TO)

    color = (0, 0, 255)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, RESIZE_TO)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float).to(DEVICE)
            image = torch.unsqueeze(image, 0)
            with torch.no_grad():
                outputs = model(image.to(DEVICE))

            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                pred_classes = outputs[0]['labels']

                for j, box in enumerate(draw_boxes):
                    if pred_classes[j] == 1:
                        cv2.rectangle(frame,
                                      (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])),
                                      color, 2)

            out.write(frame)

        else:
            break

    cap.release()


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)
