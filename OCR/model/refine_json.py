import numpy as np
from PIL import Image, ImageDraw

def calc_avg_margin(box_lines):
    # 박스간 간격 계산
    dist_list = []
    for box_line in box_lines:
        for i in range(1, len(box_line)):
            dist_list.append(box_line[i][0] - box_line[i-1][2])
    
    # 이상치 기준 설정
    q1 = np.percentile(dist_list, 25)    
    q3 = np.percentile(dist_list, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 이상치 제거 후 margin 평균 계산
    return np.mean([dist for dist in dist_list if lower_bound <= dist <= upper_bound])

def calc_avg_box_x(box_lines):
    avg_box_x = 0
    for box_line in box_lines:
        for box in box_line:
            avg_box_x += box[2] - box[0]
    return avg_box_x / sum([len(box_line) for box_line in box_lines])

def calc_slope(box_line):
    slope = 0
    for i in range(1, len(box_line)):
        if box_line[i][0] - box_line[i-1][2] == 0:
            slope += 0
        else:
            slope += (box_line[i][1] - box_line[i-1][1]) / (box_line[i][0] - box_line[i-1][2])
    return slope / (len(box_line) - 1)

def make_new_boxes(spaces, pre_box, slope, avg_width):
    slope = slope * 0
    new_boxes = [
        [
            pre_box[0] + avg_width,
            pre_box[1] + slope * avg_width,
            pre_box[2] + avg_width,
            pre_box[3] + slope * avg_width
        ]
    ]
    for _ in range(spaces-1):
        new_boxes.append([
            new_boxes[-1][0] + avg_width,
            new_boxes[-1][1] + slope * avg_width,
            new_boxes[-1][2] + avg_width,
            new_boxes[-1][3] + slope * avg_width
        ])
    return new_boxes

def make_spaces_by_lines(box_lines, label_lines, image_path):
    avg_box_x = calc_avg_box_x(box_lines)
    avg_margin = calc_avg_margin(box_lines)
    avg_width = avg_box_x + avg_margin
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 공백 찾기
    refined_box_lines = []
    refined_label_lines = []
    for box_line, label_line in zip(box_lines, label_lines):
        refined_box_line = []
        refined_label_line = []
        slope = calc_slope(box_line)
        
        for i, box, label in zip(range(len(box_line)), box_line, label_line):
            if i == 0:
                refined_box_line.append(box)
                refined_label_line.append(label)
                continue
            
            pre_box = box_line[i-1]
            cur_box = box_line[i]
            spaces = int((cur_box[0] - pre_box[2]) / (avg_width * 0.95))
            if spaces > 0:
                new_boxes = make_new_boxes(spaces, pre_box, slope, avg_width)
                for new_box in new_boxes:
                    refined_box_line.append(new_box)
                    refined_label_line.append(0)
                    draw.rectangle(new_box, outline='blue')
            
            refined_box_line.append(cur_box)
            refined_label_line.append(label)    
        
        refined_box_lines.append(refined_box_line)
        refined_label_lines.append(refined_label_line)
        
    image.save(image_path)
    return refined_box_lines, refined_label_lines

def get_outline(box_lines):
    # 전체 라인의 가장 왼쪽, 가장 위, 가장 오른쪽, 가장 아래 좌표를 구함
    min_x1 = min([box[0] for box_line in box_lines for box in box_line])
    min_y1 = min([box[1] for box_line in box_lines for box in box_line])
    max_x2 = max([box[2] for box_line in box_lines for box in box_line])
    max_y2 = max([box[3] for box_line in box_lines for box in box_line])
    
    outline = [
        min_x1,
        min_y1,
        max_x2,
        max_y2
    ]
    return outline

def draw_outline(image_path, outline):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle(outline, outline='black')
    image.save(image_path)

def make_new_black_boxes_left(spaces, left_box, slope, avg_width):
    slope = slope * 0
    new_boxes = [
        [
            left_box[0] - avg_width,
            left_box[1] - slope * avg_width,
            left_box[2] - avg_width,
            left_box[3] - slope * avg_width
        ]
    ]
    for _ in range(spaces-1):
        new_boxes.append([
            new_boxes[-1][0] - avg_width,
            new_boxes[-1][1] - slope * avg_width,
            new_boxes[-1][2] - avg_width,
            new_boxes[-1][3] - slope * avg_width
        ])
    new_boxes.reverse()
    return new_boxes

def make_black_spaces_by_lines(box_lines, label_lines, image_path):
    avg_box_x = calc_avg_box_x(box_lines)
    avg_margin = calc_avg_margin(box_lines)
    avg_width = avg_box_x + avg_margin
    outline = get_outline(box_lines)
    draw_outline(image_path, outline)
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    black_box_lines = []
    black_label_lines = []
    for box_line, label_line in zip(box_lines, label_lines):
        black_box_line = []
        black_label_line = []
        slope = calc_slope(box_line)
        
        # 왼쪽
        spaces = int((box_line[0][0] - outline[0]) / (avg_width * 0.93))
        if spaces > 0:
            new_boxes = make_new_black_boxes_left(spaces, box_line[0], slope, avg_width)
            for new_box in new_boxes:
                black_box_line.append(new_box)
                # black_label_line.append(-1)
                black_label_line.append(0)
                draw.rectangle(new_box, outline='black')

        for box, label in zip(box_line, label_line):
            black_box_line.append(box)
            black_label_line.append(label)
        
        # 오른쪽
        spaces = int((outline[2] - box_line[-1][2]) / (avg_width))
        if spaces > 0:
            new_boxes = make_new_boxes(spaces, box_line[-1], slope, avg_width)
            for new_box in new_boxes:
                black_box_line.append(new_box)
                # black_label_line.append(-1)
                black_label_line.append(0)
                draw.rectangle(new_box, outline='black')
                
        black_box_lines.append(black_box_line)
        black_label_lines.append(black_label_line)            
    
    image.save(image_path)
    
    return black_box_lines, black_label_lines
    

def labels_to_brl(label_lines):
    brl_lines = []
    for label_line in label_lines:
        brl = ""
        for label in label_line:
            brl += chr(label + 0x2800)
        brl_lines.append(brl)
    return brl_lines
    
    
def main(json_result, boxes, labels):
    # 시간 측정
    import time
    start = time.time()
    print("refine json start", start)
    
    image_path = json_result['image_path']
    refined_boxes, refined_labels = make_spaces_by_lines(boxes, labels, image_path)
    result_boxes, result_labels = make_black_spaces_by_lines(refined_boxes, refined_labels, image_path)
    brl_lines = labels_to_brl(result_labels)
    
    json_result['prediction']['boxes'] = result_boxes
    json_result['prediction']['labels'] = result_labels
    json_result['prediction']['brl'] = brl_lines
    
    
    print("refine json end", time.time() - start)
    
    return json_result

if __name__ == '__main__':
    import json
    with open('test.json', 'r') as f:
        json_result = json.load(f)
    
    boxes = json_result['prediction']['boxes']
    labels = json_result['prediction']['labels']
    main(json_result, boxes, labels)