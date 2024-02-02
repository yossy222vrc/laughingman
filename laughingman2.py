# Python 3.9, TF 2.10.0, CUDA 11.8, cuDDN 8.9.7.29
# Model from https://hanadamaya.net/face_detection_anime2/
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import cv2
import os
import numpy as np

from modules.models import RetinaFaceModel
from modules.utils import (load_yaml, pad_input_image, recover_pad_output)

flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2.yaml',
                    'config file path')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_float('iou_th', 0.2, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')


def draw_laughingman(frame, ann, img_height, img_width, bg):
    # PNG file from https://thelaughingman2024.jp/download/logo_01.html
    png_image  = cv2.imread('img_mark_01.png', cv2.IMREAD_UNCHANGED)

    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                 int(ann[2] * img_width), int(ann[3] * img_height)
    
    w, h    = x2 - x1, y2 - y1
    x1, y1, = max(x1 - w, 0), max(y1 - h, 0)
    x2, y2  = min(x2 + w, img_width), min(y2 + h, img_height)
    w, h    = x2 - x1, y2 - y1
    
    png_image = cv2.resize(png_image, (w, h))
    
    bg[y1:y2, x1:x2] = bg[y1:y2, x1:x2] * (1 - png_image[:, :, 3:] / 255) + \
                          png_image[:, :, :3] * (png_image[:, :, 3:] / 255)


def draw_bbox(img, ann, img_height, img_width):
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence
    text = "{:.4f}".format(ann[15])
    cv2.putText(img, text, (int(ann[0] * img_width), int(ann[1] * img_height)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


def limit_gpu_mem():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024*10)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def main(_argv):

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    limit_gpu_mem()

    # Camera init
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    
    # TF init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)

    cfg = load_yaml(FLAGS.cfg_path)

    # TF define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # TF load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    # Detection
    while True:

        ret, img_raw = camera.read()
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        if FLAGS.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                fy=FLAGS.down_scale_factor,
                                interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # TF pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # TF run model
        outputs = model(img[np.newaxis, ...]).numpy()

        # TF recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        out = img_raw
        
        # Change to True to use background color filling
        if False:
            bg = np.zeros((img_height_raw, img_width_raw, 3), dtype=np.uint8)
            bg.fill(0)
            out = bg
        
        # TF draw
        for prior_index in range(len(outputs)):
            draw_laughingman(img_raw, outputs[prior_index], 
                            img_height_raw, img_width_raw, out)

        cv2.namedWindow('camera')
        out = cv2.resize(out, (1920, 1080))
        cv2.imshow('camera', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
