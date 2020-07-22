"""Compute depth maps for images in the input folder.
"""
import os
import glob
import time
import utils

import cv2
import numpy as np
import tensorflow as tf

from transforms import Resize, NormalizeImage, PrepareForNet

def run(input_path, output_path, model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    resize_image = Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    
    def compose2(f1, f2):
        return lambda x: f2(f1(x))

    transform = compose2(resize_image, PrepareForNet())

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        input_data = np.expand_dims(img_input, axis=0)

        # compute
        # compute
        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()
            
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        prediction = cv2.resize(results, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
        utils.write_depth(filename, prediction, bits=2)
    print("finished")

if __name__ == "__main__":
    # set paths
    INPUT_PATH = "input"
    OUTPUT_PATH = "output"
    MODEL_PATH = "model-f46da743.tflite"

    # compute depth maps
    run(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)
