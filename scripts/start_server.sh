#!/bin/bash
set -e

# 애플리케이션 디렉토리로 이동
cd /home/ubuntu/app

# 가상환경 활성화
source venv/bin/activate

# 기존 프로세스 종료 (run_app.py 실행 중인 프로세스)
if pgrep -f "run_app.py"; then
    pkill -f "run_app.py"
fi

# 애플리케이션 실행
nohup python run_app.py --host=0.0.0.0 --port=80 > /home/ubuntu/app/app.log 2>&1 &
