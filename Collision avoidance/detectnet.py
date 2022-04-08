#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import cv2
import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video output object 
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
	
# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources
#input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
input = cv2.VideoCapture(0)


# process frames until the user exits
while True:
	# capture the next image
	#img = input.Capture()
	ret, img = input.read()
	img = cv2.resize(img, (700, 500))
	img1 = cv2.cvtColor (img, cv2.COLOR_BGR2RGBA)
	img = jetson.utils.cudaFromNumpy(img1)



	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	# print("detected {:d} objects in image".format(len(detections)))
	# print("heloooooooooooooooooooo")

	for detection in detections:
		# print(detection)
		#print(detections[detection].ClassID)
		top = detection.Top
		left = detection.Left
		right = detection.Right
		bottom = detection.Bottom
		classid = detection.ClassID
		jetson.utils.cudaDrawCircle(img, (left,top), 8, (255,0,0,200))
		jetson.utils.cudaDrawCircle(img, (right,bottom), 8, (255,0,0,200))
		if ((left > 150 and left < 500) and top > 300) or ((right > 150 and right < 500) and bottom > 300):
			print(' {:d} is in danger zone')
		if ((left > 70 and left < 600) and top > 380) or ((right > 70 and right < 600) and bottom > 380):
			print('{:d} is in danger zone')
	
		# print(top)
		# print(classid)



    # https://github.com/NVIDIA/DIGITS/issues/2214
	# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md
    # cudaDrawCircle(input, (cx,cy), radius, (r,g,b,a), output=None)
	# jetson.utils.cudaDrawCircle(img, (50,50), 25, (0,255,127,200))
	# cudaDrawLine(input, (x1,y1), (x2,y2), (r,g,b,a), line_width, output=None)
	jetson.utils.cudaDrawLine(img, (150,300),(500,300), (0,255,0,200), 3)
	jetson.utils.cudaDrawLine(img, (70,380),(600,380), (0,255,0,100), 3)
	 
	# render the imagerender the imagerender the imagere
	output.Render(img)
	# cv2.imshow("img1", img1)



	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
	# cv2.waitKey(0)

	# print out performance info
	#net.PrintProfilerTimes()

	# exit on input/output EOS
	# if not input.IsStreaming() or not output.IsStreaming():
	# 	break




