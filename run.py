import OCR.run_ocr_app as run_ocr_app
import logging
from flask import Flask, request, jsonify
from flasgger import Swagger

# Flask 애플리케이션 설정
app = Flask(__name__)
swagger = Swagger(app)

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

def proccess_OCR(image_file):
    logging.info("Process: OCR")
    try:
        boxes, brl_lines, image_url = run_ocr_app.run_ocr(image_file)
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
    app.run(host='0.0.0.0', port=5000, debug=False)