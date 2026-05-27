"""
PCB 이상 탐지 - EfficientAD 추론(Inference) 스크립트
anomalib v2 기반 | CPU 환경 최적화

기능:
  - 단일 이미지 / 폴더 일괄 추론
  - Anomaly Score + Fail/Pass 판정
  - Anomaly Map (히트맵) 시각화 및 저장
  - 불량 위치 바운딩박스 추출
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# ─────────────────────────────────────────────
# 설정값 (본인 환경에 맞게 수정)
# ─────────────────────────────────────────────
CKPT_PATH    = "./results/pcb/EfficientAd/v0/weights/lightning/model.ckpt"
INPUT_PATH   = "./test_images"       # 이미지 파일 또는 폴더 경로
OUTPUT_DIR   = "./inference_results" # 결과 저장 경로

IMAGE_SIZE   = (256, 256)            # 학습 시 사용한 크기와 동일하게 설정
THRESHOLD    = 0.5                   # Anomaly Score 임계값 (0~1, 조정 가능)

# 히트맵 시각화 설정
HEATMAP_ALPHA   = 0.5                # 히트맵 투명도 (0: 원본만, 1: 히트맵만)
SAVE_HEATMAP    = True               # 히트맵 이미지 저장 여부
SAVE_BBOX       = True               # 불량 위치 바운딩박스 표시 여부
BBOX_THRESHOLD  = 0.5                # 바운딩박스 추출용 이진화 임계값 (0~1)
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_image_paths(input_path: str) -> list[Path]:
    """이미지 경로 목록 반환 (파일 또는 폴더)"""
    p = Path(input_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    if p.is_file() and p.suffix.lower() in exts:
        return [p]
    elif p.is_dir():
        return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])
    else:
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {input_path}")


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    """Anomaly Map을 0~1로 정규화"""
    mn, mx = anomaly_map.min(), anomaly_map.max()
    if mx - mn < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - mn) / (mx - mn)).astype(np.float32)


def draw_heatmap(orig_bgr: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """원본 이미지에 히트맵 오버레이"""
    norm_map = normalize_map(anomaly_map)
    heatmap  = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap  = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))
    overlay  = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def draw_defect_bbox(image: np.ndarray, anomaly_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Anomaly Map에서 임계값 이상 영역을 바운딩박스로 표시
    불량 위치를 직관적으로 파악 가능
    """
    norm_map   = normalize_map(anomaly_map)
    resized    = cv2.resize(norm_map, (image.shape[1], image.shape[0]))
    binary     = (resized > threshold).astype(np.uint8) * 255

    # 모폴로지 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    defect_regions = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:   # 너무 작은 영역 무시
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        defect_regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
        # 빨간 바운딩박스 + 레이블
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(result, "DEFECT", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return result, defect_regions


def save_result_image(
    image_path: Path,
    orig_bgr: np.ndarray,
    anomaly_map: np.ndarray,
    pred_score: float,
    pred_label: int,  # 0: Normal, 1: Anomalous
    output_dir: Path,
    threshold: float,
):
    """결과 이미지 저장 (원본 | 히트맵 | 바운딩박스 3단 비교)"""
    status = "FAIL" if pred_label == 1 else "PASS"
    color  = (0, 0, 255) if pred_label == 1 else (0, 200, 0)

    # ── 히트맵 오버레이
    heatmap_img = draw_heatmap(orig_bgr, anomaly_map, alpha=HEATMAP_ALPHA)

    # ── 바운딩박스
    bbox_img, defect_regions = draw_defect_bbox(
        orig_bgr, anomaly_map, threshold=BBOX_THRESHOLD
    )

    # ── 공통 헤더 텍스트 추가
    def add_header(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return out

    h, w = orig_bgr.shape[:2]
    panel_orig  = add_header(orig_bgr,   f"Original  | {status}  Score:{pred_score:.3f}")
    panel_heat  = add_header(heatmap_img, f"Heatmap   | {status}  Score:{pred_score:.3f}")
    panel_bbox  = add_header(bbox_img,    f"BBox({len(defect_regions)}) | {status}  Score:{pred_score:.3f}")

    # ── 3분할 비교 이미지
    combined = np.hstack([panel_orig, panel_heat, panel_bbox])

    out_subdir = output_dir / status
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / f"{image_path.stem}_result{image_path.suffix}"
    cv2.imwrite(str(out_path), combined)

    return defect_regions, out_path


def run_inference(input_path: str, ckpt_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(input_path)
    logger.info(f"추론 대상 이미지 수: {len(image_paths)}장")

    # ── 모델 로드
    logger.info(f"체크포인트 로드: {ckpt_path}")
    model = EfficientAd.load_from_checkpoint(ckpt_path)
    model.eval()

    # ── Engine (CPU)
    engine = Engine(accelerator="cpu", devices=1, default_root_dir=str(output_dir))

    # ── Dataset
    dataset = PredictDataset(
        path       = Path(input_path),
        image_size = IMAGE_SIZE,
    )

    # ── 추론 실행
    logger.info("추론 시작...")
    predictions = engine.predict(
        model   = model,
        dataset = dataset,
        ckpt_path = ckpt_path,
    )

    # ── 결과 처리
    summary_rows = []

    if predictions is None:
        logger.warning("추론 결과가 없습니다.")
        return

    for pred in predictions:
        # pred는 ImageBatch 또는 list[ImageBatch]일 수 있음
        items = pred if isinstance(pred, list) else [pred]

        for item in items:
            img_path    = Path(item.image_path[0]) if isinstance(item.image_path, list) else Path(item.image_path)
            pred_score  = float(item.pred_score[0]) if item.pred_score.ndim > 0 else float(item.pred_score)
            pred_label  = int(item.pred_label[0])   if item.pred_label.ndim > 0 else int(item.pred_label)
            anomaly_map = item.anomaly_map[0].cpu().numpy() if item.anomaly_map is not None else None

            status = "FAIL" if pred_label == 1 else "PASS"
            logger.info(f"  [{status}] {img_path.name}  score={pred_score:.4f}")

            # 원본 이미지 로드
            orig_bgr = cv2.imread(str(img_path))
            if orig_bgr is None:
                logger.warning(f"  이미지 로드 실패: {img_path}")
                continue

            defect_regions = []
            if anomaly_map is not None and (SAVE_HEATMAP or SAVE_BBOX):
                defect_regions, out_path = save_result_image(
                    image_path  = img_path,
                    orig_bgr    = orig_bgr,
                    anomaly_map = anomaly_map,
                    pred_score  = pred_score,
                    pred_label  = pred_label,
                    output_dir  = output_dir,
                    threshold   = THRESHOLD,
                )

            summary_rows.append({
                "image"         : img_path.name,
                "status"        : status,
                "score"         : round(pred_score, 4),
                "defect_count"  : len(defect_regions),
                "defect_regions": defect_regions,
            })

    # ── 요약 출력
    print_summary(summary_rows)

    # ── CSV 저장
    save_csv(summary_rows, output_dir)


def print_summary(rows: list[dict]):
    total = len(rows)
    fails = sum(1 for r in rows if r["status"] == "FAIL")
    passes = total - fails

    logger.info("=" * 60)
    logger.info(f"  추론 완료  |  전체: {total}장  |  PASS: {passes}장  |  FAIL: {fails}장")
    logger.info("─" * 60)
    logger.info(f"  {'이미지':<30} {'판정':<6} {'점수':<8} {'불량위치수'}")
    logger.info("─" * 60)
    for r in rows:
        logger.info(f"  {r['image']:<30} {r['status']:<6} {r['score']:<8.4f} {r['defect_count']}개소")
    logger.info("=" * 60)


def save_csv(rows: list[dict], output_dir: Path):
    """결과를 CSV로 저장"""
    import csv
    csv_path = output_dir / "inference_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "status", "score", "defect_count", "defect_regions"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"결과 CSV 저장: {csv_path}")


# ─────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB EfficientAD 추론")
    parser.add_argument("--input",  default=INPUT_PATH,  help="이미지 경로 또는 폴더")
    parser.add_argument("--ckpt",   default=CKPT_PATH,   help="체크포인트 경로 (.ckpt)")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="결과 저장 폴더")
    args = parser.parse_args()

    run_inference(
        input_path  = args.input,
        ckpt_path   = args.ckpt,
        output_dir  = args.output,
    )
