import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# -----------------------------
# Transformer 기반 Re-ID Matching 모듈
# -----------------------------
class ReIDTransformer(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # Transformer Encoder Layer (Self-Attention 기반 특징 강화)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,     # 입력 feature 차원 (DeepSORT의 feature 크기와 맞춤)
            nhead=num_heads,     # Attention head 개수
            batch_first=True     # [batch, seq_len, emb_dim] 입력 형식 사용
        )
        # 여러 층의 Encoder 쌓기 (num_layers 만큼)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 최종 출력 차원 정규화를 위한 Linear Layer
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, emb_seq):
        """
        emb_seq: 입력 Re-ID feature 시퀀스 [batch, seq_len, emb_dim]
        """
        encoded = self.encoder(emb_seq)   # Self-Attention으로 temporal context 반영
        pooled = encoded.mean(dim=1)      # 시퀀스 차원 평균 풀링 (Temporal aggregation)
        return F.normalize(self.fc(pooled), dim=-1)  # L2 정규화된 embedding 반환


# -----------------------------
# YOLO + DeepSORT + MTMC 통합 파이프라인
# -----------------------------

# 1. 객체 탐지 모델 (YOLOv8) 로드
yolo_model = YOLO("yolov8n.pt")  # COCO 데이터셋 사전 학습 모델 사용 (사람 class 포함)

# 2. 단일 카메라 추적기 (DeepSORT) 초기화
tracker = DeepSort(max_age=30, n_init=3)

# 3. ReID Transformer 초기화
reid_transformer = ReIDTransformer()

# 4. 카메라 스트림 등록 (예시: 2개 카메라)
cams = {
    "cam1": cv2.VideoCapture("cam1.mp4"),
    "cam2": cv2.VideoCapture("cam2.mp4"),
}

# 5. 글로벌 ID 데이터베이스 (카메라 간 ID 일치 여부 저장)
global_db = {}    # {global_id: embedding}
next_global_id = 0  # 새로운 ID가 나올 때마다 증가

# -----------------------------
# 메인 루프: 각 프레임 처리
# -----------------------------
while True:
    for cam_id, cap in cams.items():
        ret, frame = cap.read()
        if not ret: 
            continue  # 비디오 끝나면 skip

        # (A) 객체 탐지: YOLO로 사람 검출
        results = yolo_model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            conf = float(box.conf[0])               # confidence score
            cls = int(box.cls[0])                   # 클래스 (0 = person)

            if cls == 0 and conf > 0.5:             # 사람만 추적, 신뢰도 > 0.5
                detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))

        # (B) 트래킹: DeepSORT로 단일 카메라 내 ID 유지
        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed(): 
                continue  # 아직 확정되지 않은 트랙은 무시

            local_id = t.track_id        # 로컬 카메라 ID
            bbox = t.to_ltrb()           # 바운딩 박스 [left, top, right, bottom]

            # (C) Transformer 기반 Re-ID 특징 추출
            # DeepSORT가 제공하는 feature (128-dim) → Transformer 입력으로 변환
            emb = torch.tensor(t.features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
            # emb shape: [1, 1, 128]
            emb_vec = reid_transformer(emb).detach().numpy()

            # (D) 글로벌 ID 매칭: 기존 DB와 cosine similarity 비교
            assigned_id = None
            best_sim = 0.0
            for gid, ref_emb in global_db.items():
                sim = float(np.dot(emb_vec, ref_emb.T))  # Cosine similarity
                if sim > 0.8 and sim > best_sim:         # 임계값(0.8) 이상일 때만 동일 인물로 간주
                    assigned_id, best_sim = gid, sim

            # (E) 새로운 글로벌 ID 할당
            if assigned_id is None:
                assigned_id = next_global_id
                global_db[assigned_id] = emb_vec
                next_global_id += 1

            # (F) 시각화: 영상에 바운딩 박스와 글로벌 ID 표시
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"GlobalID {assigned_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # (G) 프레임 출력
        cv2.imshow(cam_id, frame)

    # 'q' 입력 시 루프 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# 자원 정리
# -----------------------------
for cap in cams.values():
    cap.release()
cv2.destroyAllWindows()




