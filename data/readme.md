# process_arkit_data.py

### 사용법
코드 작성 시 경로 이상하게 설정해서 data 폴더로 이동 후 아래 명령어 실행

`python process_arkit_data.py --expname data_name` 

data_name엔 data/arkit/ 하위에 원하는 데이터 넣어 놓고 실행.

1. m4v 파일에서 image frame 추출 후 ARpose.txt 와 sync 맞춰 SyncedPose.txt 파일 생성 
2. [right,up,back] --> [right,forward,up]로 축 변환 후 key frame selection 과정  (frame 파일 마다 저장 포맷 다름)
3. select된 키 프레임에서 train : val = 0.9 : 0.1 비율로 이미지와 포즈 저장. 순서서대로 [0:0.9], [0.9:1] 나눔
4. test는 sync 맞춤 데이터에서 랜덤 추출
5.  `iphone_train_val_images` 폴더는 train+val 합쳐진 데이터

### For each target, provide relevant utilities to evaluate our system.

- process_arkit_data.py : [right,up,back] 원본대로 로드 후 키프레임 셀랙 할때만 [right,forward,up]로 진행하고 pose 저장은  [right,up,back]
추출할 데이터(arkit)
train : val : test = 0.9 : 0.1(4개만씀 blender옵션에서) : 0.2  비율은 바로 조절 가능
train,val 데이터는 select keyframe에서 비율 맞춰서 추출
test data는 싱크 맞춘 데이터에서 개수 맞춰 추출

- process_arkit_data_frame2.py :[right,up,back] process_arkit_data_frame2 거친후  [right,forward,up]
- 추출할 데이터(arkit)
train : val : test = 0.9 : 0.1(4개만씀 blender옵션에서) : 0.2  비율은 바로 조절 가능
train,val 데이터는 select keyframe에서 비율 맞춰서 추출
test data는 싱크 맞춘 데이터에서 개수 맞춰 추출

- process_arkit_data_frame3.py : [right,up,back] 그대로 저장하되 keframe select은 every 30 frames for opti-track (선택 프레임 주기 변)
데이터를 every n frame 마다 추출

#### process_arkit_frame 설명
- `Frame.tx`  : 30Hz (이미지가 30hz)  
- `ARpose.txt` : 60Hz
- `SyncedPoses.txt` : 30 Hz, 이미지와 포즈의 sync를 맞추고 저장되는 이미지와 포즈 정보

 `- process_arkit_data.py ` 의 key frame select :  
 SyncedPoses에서 포즈 정보에서 min_angle_keyframe,min_distance_keyframe 값 넘는 포즈만 셀랙
 
 `- process_arkit_data3.py `의 key frame select :
 경우 코드의 목적이 optitrack에서 취득한 포즈와 비교하기 위해 N frame 마다 key frame select을 진행.
 
 일단 optitrack을 10Hz로 포즈 정보 추출하고 
 ios_logger는 30Hz syncedpose에서 30 frame마다 key frame select해서 1Hz로 만들고 
 이 ios_logger key frame pose의 timestamp와 opitrack의 time 스탬프 비교해 
 가장 가까운 시간의 optitrack pose 정보 저장하는 코드 작상해야해



### Dataset format
- {train,val,test}_transformation.txt : `timestamp {train,val,test}/image_name r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz`
- SyncedPoses.txt : `timestamp imagenum(string) tx ty tz(m) qx qy qz qw`
[ios_logger](https://github.com/Varvrar/ios_logger) 참고



### icp
opti_transforms_train.txt vs transforms_train.txt
- opti :  timestamp r11 r12 r13 x r21 r22 r23 y r31 r32 r33 z
- arkit : timestamp imagename r11 r12 r13 x r21 r22 r23 y r31 r32 r33 z
- 기준 arkit, opti를 arkit 기준으로 
