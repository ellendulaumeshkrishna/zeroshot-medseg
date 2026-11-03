import numpy as np
import pytest

from zsmedseg.evaluation import SegmentationMeter, format_summary


def _square(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_perfect_prediction():
    meter = SegmentationMeter()
    gt = _square((10, 10), 2, 6, 2, 6)
    meter.update("a", gt.copy(), gt)
    s = meter.summary()
    assert s["target_iou"] == pytest.approx(100.0)
    assert s["target_dice"] == pytest.approx(100.0)
    assert s["per_image_dice_mean"] == pytest.approx(100.0)


def test_aggregate_differs_from_per_image_mean():
    meter = SegmentationMeter()
    big = _square((20, 20), 0, 10, 0, 10)  # 100 px, predicted perfectly
    small = _square((20, 20), 0, 2, 0, 2)  # 4 px, missed entirely
    meter.update("big", big.copy(), big)
    meter.update("small", np.zeros((20, 20), bool), small)
    s = meter.summary()
    # aggregate: I=100, P=100, G=104 -> IoU 100/104
    assert s["target_iou"] == pytest.approx(100 * 100 / 104)
    # per-image: Dice 100 and 0 -> mean 50
    assert s["per_image_dice_mean"] == pytest.approx(50.0)


def test_background_iou_and_empty_images():
    meter = SegmentationMeter()
    empty_gt = np.zeros((10, 10), bool)
    meter.update("normal", _square((10, 10), 0, 5, 0, 2), empty_gt)  # 10 false-positive px
    s = meter.summary()
    assert s["num_empty_images"] == 1 and s["empty_images_with_mask"] == 1
    assert s["target_iou"] == 0.0
    assert s["background_iou"] == pytest.approx(90.0)  # 90 correct bg px of 100
    assert "received a mask: 1 / 1" in format_summary(s)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        SegmentationMeter().update("x", np.zeros((3, 3), bool), np.zeros((4, 4), bool))


def test_nsd_matches_known_values():
    pytest.importorskip("surface_distance")
    from zsmedseg.evaluation.metrics import normalized_surface_dice

    gt = _square((40, 40), 10, 20, 10, 20)
    assert normalized_surface_dice(gt.copy(), gt, 2) == pytest.approx(100.0)
    shifted = _square((40, 40), 11, 21, 10, 20)  # 1 px shift: within the 2 px tolerance
    assert normalized_surface_dice(shifted, gt, 2) == pytest.approx(100.0)
    far = _square((40, 40), 25, 35, 25, 35)  # disjoint and far away
    assert normalized_surface_dice(far, gt, 2) == pytest.approx(0.0)
    assert normalized_surface_dice(np.zeros_like(gt), gt, 2) == pytest.approx(0.0)


def test_meter_reports_nsd():
    pytest.importorskip("surface_distance")
    meter = SegmentationMeter(nsd_tolerance=2)
    gt = _square((30, 30), 5, 15, 5, 15)
    meter.update("a", gt.copy(), gt)
    s = meter.summary()
    assert s["per_image_nsd_mean"] == pytest.approx(100.0) and s["nsd_tolerance"] == 2
    assert "NSD  (per image, tolerance 2 px)" in format_summary(s)
