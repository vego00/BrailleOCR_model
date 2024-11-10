import json

with open("test.json", "r") as f:
    json_result = json.load(f)
    
brl_cnt = 0
for line in json_result['prediction']['labels']:
    for l in line:
        if l != 0:
            brl_cnt += 1
print(brl_cnt)