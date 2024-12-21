import OCR.local_config as local_config
import OCR.model.infer_retinanet as infer_retinanet
import logging
from flask import Flask, request, jsonify
from flasgger import Swagger
import os
from pathlib import Path
import PIL.Image
import PIL.ImageOps

# Flask 애플리케이션 설정
app = Flask(__name__)
swagger = Swagger(app)

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.getLogger('PIL').setLevel(logging.WARNING)

results_dir = local_config.data_path
data_dir = Path(results_dir) / "data"
input_dir = data_dir / "input"
input_dir.mkdir(parents=True, exist_ok=True)

recognizer = infer_retinanet.BrailleInference(
        params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
        model_weights_fn=os.path.join(local_config.data_path, 'weights', 'model.t7'),
        create_script=None
    )

def proccess_OCR(image_file):
    logging.info("Process: OCR")
    try:
        boxes, brl_lines, image_url = recognizer.run_and_save(image_file)
        return jsonify({'boxes': boxes, 'brl': brl_lines, 'image_url': image_url}), 200
    except KeyError as e:
        logging.error('키 오류 발생', exc_info=True)
        return jsonify({'error': '키 오류 발생'}), 500
    except Exception as e:
        logging.error('서버 오류 발생', exc_info=True)
        return jsonify({'error': '서버 오류 발생'}), 500

@app.route('/ocr', methods=['POST'])
def OCR():
    """
    OCR API
    ---
    tags:
      - OCR
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: 이미지 파일
    responses:
      200:
        description: 성공
        schema:
          type: object
          properties:
            boxes:
              type: array
              items:
                type: array
                items:
                  type: number
            brl:
              type: array
              items:
                type: array
                items:
                  type: string
            image_url:
              type: string
              description: 이미지 S3 URL
      500:
        description: 서버 오류
        schema:
          type: object
          properties:
            error:
              type: string
    """
    return proccess_OCR(request.files['image'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)