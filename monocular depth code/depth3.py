# This is version 3 of code
# Which is using Optimized object detection algorithm using CUDA and TensorRT
# However depth is done using model in .onnx which can be converted to TensorRT
# The Result of this code is yet to be tested on Jetson AGX Xavier
# Then Version 4 will be written
#


import numpy as np
from matplotlib import patches
import time
import jetson.inference
import jetson.utils
import cv2
import argparse
import sys
import time

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=jetson.inference.detectNet.Usage() +
                                        jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video output object
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv + is_headless)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)




path_model = "models/"
# Read Network
model_name = "model-f6b98070.onnx";
# model_name = "model-small.onnx";


# Load the DNN model
model = cv2.dnn.readNet(path_model + model_name)

if (model.empty()):
    print("Could not load the neural net! - Check path")


# Set backend and target to CUDA to use GPU
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def depth_to_distance(depth):
    return -1.7 * depth + 2


# create video sources
#input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
input = cv2.VideoCapture(0)




# process frames until the user exits
while True:

	#img = input.Capture()
	ret, img = input.read()
	img = cv2.resize(img, (700, 500))
	img1 = cv2.cvtColor (img, cv2.COLOR_BGR2RGBA)
	img = jetson.utils.cudaFromNumpy(img1)
    imgHeight, imgWidth, channels = img.shape

    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=opt.overlay)

    # Depth map from neural net
    blob = cv2.dnn.blobFromImage(img, 1 / 255., (384, 384), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    depth_map = model.forward()
    depth_map = depth_map[0, :, :]
    depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))

    # Normalize the output
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    for detection in detections:
        # print(detection)
        # print(detections[detection].ClassID)
        top = detection.Top
        left = detection.Left
        right = detection.Right
        bottom = detection.Bottom
        classid = detection.ClassID
        jetson.utils.cudaDrawCircle(img, (left, top), 8, (255, 0, 0, 200))
        jetson.utils.cudaDrawCircle(img, (right, bottom), 8, (255, 0, 0, 200))
        t = int(top)
        l = int(left)
        r = int(right)
        b = int(bottom)

        width = b-t
        height = r-l

        boundBox = int(t * w), int(l * h), int(imgWidth * w), int(imgHeight * h)
        center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        # Depth
        depth = depth_map[int(center_point[1]), int(center_point[0])]
        depth = depth_to_distance(depth)
        print("depth in m is ")
        print(str(round(depth, 2)))




# render the imagerender the imagerender the imagere
output.Render(img)
# cv2.imshow("img1", img1)


# update the title bar
output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
# cv2.waitKey(0)

# print out performance info
# net.PrintProfilerTimes()

# exit on input/output EOS
# if not input.IsStreaming() or not output.IsStreaming():
# 	break

