"""
PCB 이상 탐지 - EfficientAD 학습 스크립트
anomalib v2 기반 | CPU 환경 최적화

[데이터셋 폴더 구조]
data/pcb/
├── train/
│   └── good/          ← 정상 PCB 이미지 (학습용)
│       ├── img_001.jpg
│       ├── img_002.jpg
│       └── ...
└── test/
    ├── good/          ← 정상 PCB 이미지 (평가용)
    │   └── ...
    └── defect/        ← 불량 PCB 이미지 (평가용, 선택사항)
        └── ...
"""

import logging
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# ─────────────────────────────────────────────
# 설정값 (본인 환경에 맞게 수정)
# ─────────────────────────────────────────────
DATA_ROOT        = "./data/pcb"          # 데이터 루트 경로
NORMAL_DIR       = "train/good"          # 정상 이미지 경로 (루트 기준)
ABNORMAL_DIR     = "test/defect"         # 불량 이미지 경로 (없으면 None으로 설정)
NORMAL_TEST_DIR  = "test/good"           # 테스트용 정상 이미지
RESULT_DIR       = "./results"           # 결과 저장 경로

IMAGE_SIZE       = (256, 256)            # 입력 이미지 크기 (256 권장)
TRAIN_BATCH_SIZE = 4                     # CPU 환경: 작게 설정 (4~8)
EVAL_BATCH_SIZE  = 4
NUM_WORKERS      = 0                     # CPU 환경: 0 권장 (멀티프로세싱 오버헤드 방지)
MAX_EPOCHS       = 10                    # 학습 에폭 수

# EfficientAD 모델 크기: "s" (small, 빠름) / "m" (medium, 정확)
MODEL_SIZE       = "s"

# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_datamodule() -> Folder:
    """커스텀 PCB 데이터셋 DataModule 구성"""
    abnormal_dir = ABNORMAL_DIR if Path(f"{DATA_ROOT}/{ABNORMAL_DIR}").exists() else None

    datamodule = Folder(
        name        = "pcb",
        root        = DATA_ROOT,
        normal_dir  = NORMAL_DIR,
        abnormal_dir= abnormal_dir,
        normal_test_dir = NORMAL_TEST_DIR if Path(f"{DATA_ROOT}/{NORMAL_TEST_DIR}").exists() else None,
        # abnormal_dir가 없으면 synthetic 모드로 자동 생성
        test_split_mode = "from_dir" if abnormal_dir else "synthetic",
        val_split_mode  = "same_as_test",
        val_split_ratio = 0.5,
        image_size      = IMAGE_SIZE,
        train_batch_size= TRAIN_BATCH_SIZE,
        eval_batch_size = EVAL_BATCH_SIZE,
        num_workers     = NUM_WORKERS,
        task            = "segmentation",  # anomaly map(히트맵) 생성 활성화
    )
    return datamodule


def build_model() -> EfficientAd:
    """EfficientAD 모델 구성 (CPU 최적화)"""
    model = EfficientAd(
        model_size        = MODEL_SIZE,   # "s": 경량, "m": 정확도 우선
        lr                = 1e-4,
        weight_decay      = 1e-5,
        # imagenet_dir: EfficientAD는 내부적으로 ImageNet 샘플 이미지를 사용
        # 자동 다운로드되므로 별도 설정 불필요
    )
    return model


def main():
    logger.info("=" * 55)
    logger.info(" PCB EfficientAD 학습 시작")
    logger.info("=" * 55)
    logger.info(f"  데이터 경로 : {Path(DATA_ROOT).resolve()}")
    logger.info(f"  이미지 크기 : {IMAGE_SIZE}")
    logger.info(f"  모델 크기   : EfficientAD-{MODEL_SIZE.upper()}")
    logger.info(f"  학습 에폭   : {MAX_EPOCHS}")
    logger.info(f"  디바이스    : CPU")
    logger.info("=" * 55)

    # 디바이스 강제 CPU 설정
    torch.set_num_threads(max(1, torch.get_num_threads()))

    datamodule = build_datamodule()
    model      = build_model()

    engine = Engine(
        max_epochs       = MAX_EPOCHS,
        accelerator      = "cpu",          # CPU 강제 지정
        devices          = 1,
        default_root_dir = RESULT_DIR,
        # 학습 로그 주기 (n 배치마다 출력)
        log_every_n_steps= 5,
    )

    logger.info("학습 시작...")
    engine.fit(model=model, datamodule=datamodule)

    logger.info("테스트(평가) 시작...")
    test_results = engine.test(model=model, datamodule=datamodule)

    logger.info("─" * 55)
    logger.info("평가 결과:")
    for result in test_results:
        for key, val in result.items():
            logger.info(f"  {key:<30} : {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # 체크포인트 경로 출력
    ckpt_candidates = list(Path(RESULT_DIR).rglob("*.ckpt"))
    if ckpt_candidates:
        latest_ckpt = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
        logger.info("─" * 55)
        logger.info(f"저장된 체크포인트: {latest_ckpt}")
        logger.info("→ inference.py의 CKPT_PATH에 위 경로를 입력하세요.")
    logger.info("=" * 55)
    logger.info("학습 완료!")


if __name__ == "__main__":
    main()
