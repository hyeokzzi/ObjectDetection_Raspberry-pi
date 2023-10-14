# Raspberry-Pi-AI_Project
라즈베리파이를 이용하여 신경망 학습 및 객체 탐지 활동

## 영상 처리 속도 확인을 위한 Tiny 모델 사용
Yolo2-tiny 모델을  사용하여 라즈베리파이에서 영상 처리 속도 확인
| 파일 | 상세내용 |
| --- | --- |
| Use_Tiny.py | 이미지 촬영 후  tiny 모델의 신경망을 통해 처리된 영상 송출 |
| Use_Tiny_Fps.py | 위 파일에 초당 전송 Frame 표시 추가 |

## 두개의 카메라로 Yolo 사용 실험

- 간단하게 구현하기 위해 하나의 카메라로 yolo를 구현하는 코드를 복사하여 두개의 카메라가 구현하도록 설정하였다.
- 파이캠 1개(0번,-1번)와 웹캠 1개(1번)을 출력하여 객체를 인식하도록 하였다.
  
| 파일 | 상세내용 |
| --- | --- |
| Two_camera.py | 파이캠(0번), WebCam(1번) 연결하여 동시 송출 및 객체 인식 파일 |
   
![실행결과](https://github.com/hyeokzzi/ObjectDetection_Raspberry-pi/assets/87352996/ee22ede9-6d8d-4298-a27e-4f17e7423f0d)

   
