import rospy
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import time
import jetson.inference
import jetson.utils
import cv2
import argparse
import sys

# import stereoutils



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






# Adding functions for depth estimation



def compute_left_disparity_map(img_left, img_right):


    # Parameters
    num_disparities = 6 * 16
    block_size = 11

    min_disparity = 0
    window_size = 6

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Stereo BM matcher
    left_matcher_BM = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )

    # Stereo SGBM matcher
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the left disparity map
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32) / 16



    return disp_left


def decompose_projection_matrix(p):


    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = t / t[3]



    return k, r, t


def calc_depth_map(disp_left, k_left, t_left, t_right):

    # Get the focal length from the K matrix
    f = k_left[0, 0]

    # Get the distance between the cameras from the t matrices (baseline)
    b = t_left[1] - t_right[1]

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.ones(disp_left.shape, np.single)

    # Calculate the depths
    depth_map[:] = f * b / disp_left[:]



    return depth_map


def locate_obstacle_in_image(image, obstacle_image):


    # Run the template matching from OpenCV
    cross_corr_map = cv2.matchTemplate(image, obstacle_image, method=cv2.TM_CCOEFF)

    # Locate the position of the obstacle using the minMaxLoc function from OpenCV
    _, _, _, obstacle_location = cv2.minMaxLoc(cross_corr_map)


    return cross_corr_map, obstacle_location



def calculate_nearest_point(depth_map, obstacle_location, obstacle_img):

    # Gather the relative parameters of the obstacle box
    obstacle_width = obstacle_img.shape[0]
    obstacle_height = obstacle_img.shape[1]
    obstacle_min_x_pos = obstacle_location[1]
    obstacle_max_x_pos = obstacle_location[1] + obstacle_width
    obstacle_min_y_pos = obstacle_location[0]
    obstacle_max_y_pos = obstacle_location[0] + obstacle_height

    # Get the depth of the pixels within the bounds of the obstacle image, find the closest point in this rectangle
    obstacle_depth = depth_map_left[obstacle_min_x_pos:obstacle_max_x_pos, obstacle_min_y_pos:obstacle_max_y_pos]
    closest_point_depth = obstacle_depth.min()



    # Create the obstacle bounding box
    obstacle_bbox = patches.Rectangle((obstacle_min_y_pos, obstacle_min_x_pos), obstacle_height, obstacle_width,
                                      linewidth=1, edgecolor='r', facecolor='none')

    return closest_point_depth, obstacle_bbox


def get_projection_matrices():
    """Frame Calibration Holder
    3x4    p_left, p_right      Camera P matrix. Contains extrinsic and intrinsic parameters.
    """
    p_left = np.array([[640.0,   0.0, 640.0, 2176.0],
                       [  0.0, 480.0, 480.0,  552.0],
                       [  0.0,   0.0,   1.0,    1.4]])
    p_right = np.array([[640.0,   0.0, 640.0, 2176.0],
                       [   0.0, 480.0, 480.0,  792.0],
                       [   0.0,   0.0,   1.0,    1.4]])
    return p_left, p_right




# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0


# process frames until the user exits
while True:
	img_left = cv2.imread("/media/brain/Data/jetson-inference/python/examples/stereo/frame_00077_1547042741L.png")
	img_right = cv2.imread("/media/brain/Data/jetson-inference/python/examples/stereo/frame_00077_1547042741R.png")
	p_left, p_right = get_projection_matrices()


	# img = cv2.resize(img_left, (700, 500))
	img1 = cv2.cvtColor (img_left, cv2.COLOR_BGR2RGBA)
	img = jetson.utils.cudaFromNumpy(img1)



	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	# print("detected {:d} objects in image".format(len(detections)))
	# print("heloooooooooooooooooooo")

	for detection in detections:
		disp_left = compute_left_disparity_map(img_left, img_right)
		k_left, r_left, t_left = decompose_projection_matrix(p_left)
		k_right, r_right, t_right = decompose_projection_matrix(p_right)
		depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)
		

		# print(detection)
		#print(detections[detection].ClassID)
		top = detection.Top
		left = detection.Left
		right = detection.Right
		bottom = detection.Bottom
		classid = detection.ClassID
		jetson.utils.cudaDrawCircle(img, (left,top), 8, (255,0,0,200))
		jetson.utils.cudaDrawCircle(img, (right,bottom), 8, (255,0,0,200))
		t= int(top)
		l= int(left)
		r= int(right)
		b= int(bottom)



		img_left_colour = img_left
		obstacle_image = img_left_colour[t:l, b:r, :]

		cross_corr_map, obstacle_location = locate_obstacle_in_image(img_left, obstacle_image)
		closest_point_depth, obstacle_bbox = calculate_nearest_point(depth_map_left, obstacle_location, obstacle_image)

		# Print Result Output
		print("Left Projection Matrix Decomposition:\n {0}".format([k_left.tolist(),
                                                            r_left.tolist(),
                                                            t_left.tolist()]))
		print("\nRight Projection Matrix Decomposition:\n {0}".format([k_right.tolist(),
                                                               r_right.tolist(),
                                                               t_right.tolist()]))
		print("\nObstacle Location (left-top corner coordinates):\n {0}".format(list(obstacle_location)))
		print("\nClosest point depth of " + str(classid) + " (meters):\n {0}".format(closest_point_depth))

		
		# print(top)
		print(classid)
	 
	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time)
	prev_frame_time = new_frame_time
 
    # converting the fps into integer
	fps = int(fps)
	print(fps)
	# render the imagerender the imagerender the imagere
	output.Render(img)
	# cv2.imshow("img1", img1)


	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
	# cv2.waitKey(0)












