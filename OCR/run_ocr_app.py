#!/usr/bin/env python
# coding: utf-8
"""
Local application for Angelina Braille Reader inference
"""
import os
import boto3

from pathlib import Path
import PIL.Image

import OCR.local_config as local_config
import OCR.model.infer_retinanet as infer_retinanet

aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
bucket_name = os.environ['BUCKET_NAME']
file_key = os.environ['FILE_KEY']

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='ap-northeast-2'
)

# model_weights = 'model.t7'
local_model_path = os.path.join(local_config.data_path, 'OCR', 'weights', 'model_weights.pth')

with open(local_model_path, 'wb') as f:
    s3.download_fileobj(bucket_name, file_key, f)

recognizer = infer_retinanet.BrailleInference(
    params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
    model_weights_fn=local_model_path,
    create_script=None
)
def run_ocr(image):
    results_dir = local_config.data_path

    recognizer.run_and_save(image, results_dir, target_stem=None,
                                               lang='RU', extra_info=None,
                                               draw_refined=recognizer.DRAW_NONE,
                                               remove_labeled_from_filename=False,
                                               find_orientation=False,
                                               align_results=True,
                                               process_2_sides=False,
                                               repeat_on_aligned=False,
                                               save_development_info=False)

    # print("result: ", recognizer.get_result())
    return recognizer.get_result()