import argparse
import os

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import inception_resnet_v1

def estimate_age_and_gender(aligned_images, model_path):
    with tf.Graph().as_default():
        session = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        session.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            pass
        return session.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})


def detect_face(image_path, shape_predictor):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    faceAligner = FaceAligner(predictor, desiredFaceWidth=160)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles = detector(gray, 1)

    num_rectangles = len(rectangles)
    XY, aligned_images = [], []
    if num_rectangles == 0:
        aligned_images.append(image)
        return aligned_images, image, XY
    else:
        for i in range(num_rectangles):
            aligned_image = faceAligner.align(image, gray, rectangles[i])
            aligned_images.append(aligned_image)
            (x, y, w, h) = rect_to_bb(rectangles[i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            XY.append((x, y))
        return np.array(aligned_images), image, XY


def draw_label(image, point, ages, genders,font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    for i in range(len(point)):
        label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point[i]
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point[i], font, font_scale, (255, 255, 255), thickness)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--image_path", "--I", required=True, type=str, help="Image Path")
    argument_parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")
    argument_parser.add_argument("--shape_detector", "--S", default="shape_predictor_68_face_landmarks.dat", type=str,
                        help="Shape Detector Path")

    argument_parser.add_argument("--font_scale", type=int, default=1, help="Control font size of text on picture.")
    argument_parser.add_argument("--thickness", type=int, default=1, help="Control thickness of texton picture.")
    args = argument_parser.parse_args()

    image_path = args.image_path
    shape_detector = args.shape_detector

    aligned_image, image, XY = detect_face(image_path, shape_detector)

    model_path = args.model_path

    ages, genders = estimate_age_and_gender(aligned_image, model_path)

    draw_label(image, XY, ages, genders, font_scale=args.font_scale, thickness=args.thickness)
    plt.imshow(image[:, :, (2, 1, 0)])
    plt.show()
