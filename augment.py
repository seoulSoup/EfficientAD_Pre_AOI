"""
PCB 이미지 리사이즈 + Augmentation 스크립트
- 입력 폴더의 이미지를 resize 후 밝기/대비 augmentation 적용
- 출력 이미지 수를 TARGET_COUNT로 지정 가능
  예) 입력 50장, TARGET_COUNT=200 → 원본 50장 + augmented 150장 출력
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
INPUT_DIR    = "./data/original"       # 원본 이미지 폴더
OUTPUT_DIR   = "./data/augmented"      # 출력 폴더
TARGET_SIZE  = (256, 256)              # 목표 크기 (W, H)

# 리사이즈 방식
# "stretch" : 비율 무시, 256x256으로 압축 (PCB 정렬된 경우 권장)
# "fit_pad" : 비율 유지 + 패딩
RESIZE_MODE  = "stretch"
PAD_COLOR    = (114, 114, 114)
INTERPOLATION = cv2.INTER_AREA

# ── 출력 이미지 수 설정 ──────────────────────
# None  : 원본 장수만큼만 (augmentation 없이 resize만)
# 정수  : 원본 포함해서 총 N장 출력
#         예) 원본 50장 → TARGET_COUNT=200이면 augmented 150장 추가 생성
TARGET_COUNT = 200

# ── Augmentation 파라미터 ────────────────────
# 밝기 조정: 픽셀값에 [-BRIGHTNESS_DELTA, +BRIGHTNESS_DELTA] 범위 랜덤 가산
BRIGHTNESS_DELTA = 30       # 0~255 기준, 클수록 밝기 변화 폭 커짐

# 대비 조정: 픽셀값에 [1-CONTRAST_RANGE, 1+CONTRAST_RANGE] 범위 랜덤 배율
CONTRAST_RANGE   = 0.3      # 0.3이면 x0.7 ~ x1.3 배율

# 감마 보정: [1-GAMMA_RANGE, 1+GAMMA_RANGE] 범위 랜덤 감마
GAMMA_RANGE      = 0.2      # 0.2이면 감마 0.8 ~ 1.2

# 가우시안 노이즈: 표준편차 [0, NOISE_STD_MAX] 랜덤
NOISE_STD_MAX    = 8        # 0이면 노이즈 비활성화

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 리사이즈 ──────────────────────────────────

def resize_stretch(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, TARGET_SIZE, interpolation=INTERPOLATION)


def resize_fit_pad(img: np.ndarray) -> np.ndarray:
    tw, th = TARGET_SIZE
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=INTERPOLATION)
    top    = (th - nh) // 2
    bottom = th - nh - top
    left   = (tw - nw) // 2
    right  = tw - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=PAD_COLOR)


def do_resize(img: np.ndarray) -> np.ndarray:
    if RESIZE_MODE == "stretch":
        return resize_stretch(img)
    elif RESIZE_MODE == "fit_pad":
        return resize_fit_pad(img)
    raise ValueError(f"알 수 없는 RESIZE_MODE: {RESIZE_MODE}")


# ── Augmentation ──────────────────────────────

def aug_brightness(img: np.ndarray) -> np.ndarray:
    """밝기 랜덤 조정"""
    delta = random.uniform(-BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)
    out = img.astype(np.int16) + int(delta)
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_contrast(img: np.ndarray) -> np.ndarray:
    """대비 랜덤 조정 (평균 기준 스케일링)"""
    alpha = random.uniform(1.0 - CONTRAST_RANGE, 1.0 + CONTRAST_RANGE)
    mean  = img.mean()
    out   = (img.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_gamma(img: np.ndarray) -> np.ndarray:
    """감마 보정 랜덤 적용"""
    gamma = random.uniform(1.0 - GAMMA_RANGE, 1.0 + GAMMA_RANGE)
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def aug_noise(img: np.ndarray) -> np.ndarray:
    """가우시안 노이즈 추가"""
    if NOISE_STD_MAX <= 0:
        return img
    std = random.uniform(0, NOISE_STD_MAX)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img: np.ndarray) -> np.ndarray:
    """
    밝기 → 대비 → 감마 → 노이즈 순서로 augmentation 적용
    각 단계는 독립적으로 랜덤 강도 적용
    """
    img = aug_brightness(img)
    img = aug_contrast(img)
    img = aug_gamma(img)
    img = aug_noise(img)
    return img


# ── 메인 처리 ─────────────────────────────────

def load_images(input_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """입력 폴더에서 이미지 로드 후 resize"""
    image_paths = sorted([p for p in input_dir.rglob("*")
                          if p.suffix.lower() in EXTENSIONS])
    if not image_paths:
        raise FileNotFoundError(f"이미지 없음: {input_dir}")

    loaded = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning(f"  로드 실패 (건너뜀): {p.name}")
            continue
        resized = do_resize(img)
        loaded.append((p, resized))
        logger.info(f"  로드: {p.name}  ({img.shape[1]}x{img.shape[0]}) → {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    return loaded


def process(input_dir: str, output_dir: str):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더 없음: {input_dir}")

    logger.info("=" * 60)
    logger.info(f"  리사이즈 + Augmentation")
    logger.info(f"  입력  : {input_dir}")
    logger.info(f"  출력  : {output_dir}")
    logger.info(f"  크기  : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}  모드: {RESIZE_MODE}")
    logger.info(f"  목표 장수: {TARGET_COUNT if TARGET_COUNT else '원본만'}")
    logger.info("=" * 60)

    # 1. 이미지 로드 + resize
    loaded = load_images(input_dir)
    n_orig = len(loaded)
    if n_orig == 0:
        logger.error("처리 가능한 이미지가 없습니다.")
        return

    # 2. 원본 저장
    saved = 0
    for src_path, img in loaded:
        dst = output_dir / f"{src_path.stem}_orig.jpg"
        cv2.imwrite(str(dst), img)
        saved += 1

    logger.info(f"  원본 {n_orig}장 저장 완료")

    # 3. Augmentation으로 부족분 채우기
    if TARGET_COUNT and TARGET_COUNT > n_orig:
        n_aug = TARGET_COUNT - n_orig
        logger.info(f"  Augmentation {n_aug}장 생성 중...")

        aug_idx = 0
        while aug_idx < n_aug:
            # 원본 이미지 중 랜덤 선택
            src_path, base_img = random.choice(loaded)
            aug_img = augment(base_img)

            dst = output_dir / f"{src_path.stem}_aug{aug_idx:04d}.jpg"
            cv2.imwrite(str(dst), aug_img)
            aug_idx += 1
            saved += 1

            if aug_idx % 50 == 0 or aug_idx == n_aug:
                logger.info(f"    ... {aug_idx}/{n_aug}장 완료")

    logger.info("─" * 60)
    logger.info(f"  완료  |  원본: {n_orig}장  |  Augmented: {saved - n_orig}장  |  합계: {saved}장")
    logger.info(f"  출력 폴더: {output_dir.resolve()}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB 이미지 리사이즈 + Augmentation")
    parser.add_argument("--input",   default=INPUT_DIR,    help="원본 이미지 폴더")
    parser.add_argument("--output",  default=OUTPUT_DIR,   help="출력 폴더")
    parser.add_argument("--size",    default=256, type=int, help="목표 크기 (기본 256)")
    parser.add_argument("--mode",    default=RESIZE_MODE,
                        choices=["stretch", "fit_pad"],    help="리사이즈 방식")
    parser.add_argument("--count",   default=TARGET_COUNT, type=int,
                        help="출력 총 이미지 수 (원본 포함, 0이면 원본만)")
    args = parser.parse_args()

    TARGET_SIZE  = (args.size, args.size)
    RESIZE_MODE  = args.mode
    TARGET_COUNT = args.count if args.count > 0 else None

    process(args.input, args.output)
