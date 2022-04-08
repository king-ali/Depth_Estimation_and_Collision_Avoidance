# This is version 2 of code
# Which is to just see the monocular depth estimation result without detected object
import cv2
import time


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


cap = cv2.VideoCapture(0)


while cap.isOpened():

    success, img = cap.read()

    imgHeight, imgWidth, channels = img.shape

    start = time.time()


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # FOR testing purpose
    center_point = (0, 0)
    # center_point = TO GET POINT FROM DETECTION ALGORITHM
    print("with 0 center point")

    # Depth map from neural net

    # MiDaS ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    blob = cv2.dnn.blobFromImage(img, 1 / 255., (384, 384), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    depth_map = model.forward()
    depth_map = depth_map[0, :, :]
    depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))

    # Normalize the output
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Convert the image color back so it can be displayed
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    # Depth
    depth = depth_map[int(center_point[1]), int(center_point[0])]
    depth = depth_to_distance(depth)
    cv2.putText(img, "Depth in m: " + str(round(depth, 2)), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0), 3)

    # Depth converted to distance

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    # print("FPS: ", fps)

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('image', img)
    cv2.imshow('Depth map', depth_map)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

