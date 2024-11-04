#!/bin/bash
# 애플리케이션 중지
if pgrep -f "flask run"; then pkill -f "flask run"; fi
