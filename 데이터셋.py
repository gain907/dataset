import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['', '', '']
seq_length = 30
secs_for_action = 30
# 랜드마크 데이터 63개: 21개의 랜드마크 × 각 랜드마크의 3차원 좌표 (x, y, z) = 63개
# 손가락 각도 데이터 15개: 5개 손가락 × 각 손가락의 3개 관절 각도 = 15개
# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) # 데이터셋 저장할 폴더

while cap.isOpened():

    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)  # 반전 시켜줌
        # 어떤 액션을 취할지 텍스트 보여줌
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000) # 3초동안 기다려 주세요
        start_time = time.time()

        while time.time() - start_time < secs_for_action:  # 30초 동안 반복
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    # x, y, z 좌표와 visibility 값을 joint 배열에 저장
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]  # visibility 랜드마크 보이는지/안보이는지 0~1
                    # 손가락 관절사이 각도 구하는 공식 #################
                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree
                    ##############################################

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)  # idx = 종류 = 
                    # joint = [lm.x, lm.y, lm.z, lm.visibility]
                    d = np.concatenate([joint.flatten(), angle_label]) # 100개짜리  행렬이 됨

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        # npy형태로 저장
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
