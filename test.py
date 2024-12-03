import requests

url = 'http://3.38.213.235:5000/ocr'
files = {'image': open('data/서울사랑/서울사랑01.jpg', 'rb')}

response = requests.post(url, files=files)

print(response.status_code)
print(response.json())