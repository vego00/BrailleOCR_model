import csv
import json
from pathlib import Path

def calculate_iou(box1, box2):
    """
    두 박스 간의 IoU(Intersection over Union)를 계산
    :param box1: 첫 번째 박스 좌표 [x1, y1, x2, y2]
    :param box2: 두 번째 박스 좌표 [x1, y1, x2, y2]
    :return: IoU 값
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 교차 영역이 없으면 IoU는 0
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    # 교차 영역 넓이
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # 두 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU 계산
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def calculate_accuracy(predicted_labels, predicted_boxes, csv_file_path, json_file_path, iou_threshold=0.5):
    """
    좌표 기반의 정확도를 계산하는 함수
    :param predicted_boxes: 모델이 예측한 좌표 리스트 [[x1, y1, x2, y2], ...]
    :param csv_file_path: 정답 데이터가 있는 CSV 파일 경로 (세미콜론 구분)
    :param iou_threshold: IoU 임계값, 이 값 이상이면 인식에 성공한 것으로 간주
    :return: 인식 성공률 (정확도)
    """
    # CSV 파일에서 정답 데이터를 읽어오기
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        correct_labels = [row[4] for row in reader]

    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        data = json_data["shapes"]
    
    correct_boxes = []
    for i in range(len(data)):
        correct_boxes.append([data[i]["points"][0][0], data[i]["points"][0][1], data[i]["points"][1][0], data[i]["points"][1][1]])

    total_braille_characters = len(correct_boxes)
    recognized_braille_characters = 0

    for predicted_box, predicted_label in zip(predicted_boxes, predicted_labels):
        for i, correct_box in enumerate(correct_boxes):
            iou = calculate_iou(predicted_box, correct_box)
            if iou >= iou_threshold:
                # 매칭된 박스에 해당하는 라벨을 비교
                correct_label = correct_labels[i]
                if int(predicted_label) == int(correct_label):
                    recognized_braille_characters += 1
                
                # 매칭된 정답 박스를 제거하여 중복 검사를 방지
                del correct_boxes[i]
                del correct_labels[i]
                break
            
    # 정확도 계산
    accuracy = recognized_braille_characters / total_braille_characters
    return accuracy

def save_accuracy(results_dir, target_stem):
    csv_file_path = Path(results_dir) / (target_stem + ".csv")
    json_file_path = Path(results_dir) / (target_stem + ".json")

    # JSON 파일 경로 설정
    json_dir = Path(results_dir) / "annotations"
    json_path = json_dir / (target_stem + ".marked.json")
    
    with open(json_path, "r") as f:
        json_data = json.load(f)
        
    # accuracy 계산
    predicted_labels = json_data["prediction"]["labels"]
    predicted_boxes = json_data["prediction"]["boxes"]
    accuracy = calculate_accuracy(predicted_labels, predicted_boxes, csv_file_path, json_file_path)
    print(f"Accuracy for {target_stem}: {accuracy:.2f}")
    
    json_result = {
        "accuracy": accuracy
    }
    
    json_data.update(json_result)
    
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)