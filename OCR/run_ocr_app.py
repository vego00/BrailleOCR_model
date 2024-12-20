#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path

import OCR.local_config as local_config
import OCR.model.infer_retinanet as infer_retinanet
import PIL.Image

# try:
#     aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
# except KeyError:
#     aws_access_key_id = None
#     print('AWS_ACCESS_KEY_ID not found in environment variables')

# try:
#     aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
# except KeyError:
#     aws_secret_access_key = None
#     print('AWS_SECRET_ACCESS_KEY not found in environment variables')
    
# try:
#     bucket_name = os.environ['BUCKET_NAME']
# except KeyError:
#     bucket_name = None
#     print('BUCKET_NAME not found in environment variables')
    
# try:
#     file_key = os.environ['FILE_KEY']
# except KeyError:
#     file_key = None
#     print('FILE_KEY not found in environment variables')

# try:
#     s3 = boto3.client(
#         's3',
#         aws_access_key_id=aws_access_key_id,
#         aws_secret_access_key=aws_secret_access_key,
#         region_name='ap-northeast-2'
#     )
# except Exception as e:
#     print('Failed to create s3 client:', e)

# model_weights = 'model.t7'
# try:
#     with open(local_model_path, 'wb') as f:
#         s3.download_fileobj(bucket_name, file_key, f)
# except Exception as e:
#     print('Failed to download model weights from S3')

# recognizer = infer_retinanet.BrailleInference(
#     params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
#     model_weights_fn=os.path.join(local_config.data_path, 'weights', model_weights),
#     create_script=None
# )

def run_ocr(recognizer, image_file):
    results_dir = local_config.data_path
    # recognizer = infer_retinanet.BrailleInference(
    #     params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
    #     model_weights_fn=os.path.join(local_config.data_path, 'weights', 'model.t7'),
    #     create_script=None
    # )
    return recognizer.run_and_save(image_file, results_dir, target_stem=None,
                                               lang='RU', extra_info=None,
                                               draw_refined=recognizer.DRAW_NONE,
                                               remove_labeled_from_filename=False,
                                               find_orientation=False,
                                               align_results=True,
                                               process_2_sides=False,
                                               repeat_on_aligned=False,
                                               save_development_info=False)

    # print("result: ", recognizer.get_result())
    # return recognizer.get_result()