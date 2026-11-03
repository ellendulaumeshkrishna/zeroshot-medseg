"""Unit tests for the pipeline stages that need no model weights."""

import numpy as np
import pytest

from zsmedseg.models.medsam2 import MaskCandidate
from zsmedseg.prompting import MultiPeakPrompts
from zsmedseg.saliency.estimator import centred_padding, sliding_windows
from zsmedseg.selection import MaskSelector


def _candidate(score, prompt=0, idx=0, logits=None):
    return MaskCandidate(
        mask=np.zeros((4, 4), bool),
        score=score,
        low_res_logits=np.zeros((4, 4)) if logits is None else logits,
        prompt_index=prompt,
        mask_index=idx,
    )


# --- prompting -----------------------------------------------------------------------


def test_multi_peak_order_and_suppression():
    sal = np.zeros((100, 100), np.float32)
    sal[10, 10], sal[12, 12], sal[80, 80] = 0.9, 0.8, 0.7  # (12,12) is within the 1st square
    points = MultiPeakPrompts(num_points=3, suppression_frac=0.10)(sal)
    assert points[0] == (10, 10)
    assert (12, 12) not in points
    assert points[1] == (80, 80)


def test_multi_peak_stops_on_non_positive():
    sal = np.zeros((30, 30), np.float32)
    sal[5, 5] = 0.5
    assert MultiPeakPrompts(num_points=5)(sal) == [(5, 5)]


def test_multi_peak_does_not_modify_input():
    sal = np.random.RandomState(0).rand(20, 20).astype(np.float32)
    before = sal.copy()
    MultiPeakPrompts()(sal)
    assert np.array_equal(sal, before)


# --- selection -----------------------------------------------------------------------


def test_highest_predicted_iou_wins_and_ties_keep_first():
    cands = [_candidate(0.5, 0, 0), _candidate(0.9, 0, 1), _candidate(0.9, 1, 0)]
    chosen = MaskSelector()(cands)
    assert (chosen.prompt_index, chosen.mask_index) == (0, 1)
    assert chosen.score == pytest.approx(0.9)
    assert MaskSelector()([]) is None


# --- saliency geometry ---------------------------------------------------------------


def test_sliding_windows_cover_image():
    windows = list(sliding_windows(336, 441, 336, 112))
    assert windows == [(0, 0, 336, 336), (0, 105, 336, 441)]
    covered = np.zeros((336, 441), int)
    for y1, x1, y2, x2 in windows:
        covered[y1:y2, x1:x2] += 1
    assert covered.min() >= 1


def test_centred_padding():
    assert centred_padding(336, 336, 56) == (0, 0, 0, 0)
    assert centred_padding(300, 330, 56) == (3, 3, 18, 18)


def _mask_candidate(score, prompt, box):
    mask = np.zeros((10, 10), bool)
    y0, y1, x0, x1 = box
    mask[y0:y1, x0:x1] = True
    return MaskCandidate(mask=mask, score=score, low_res_logits=np.zeros((2, 2)), prompt_index=prompt, mask_index=0)


def test_two_objects_unions_disjoint_masks():
    left = _mask_candidate(0.9, 0, (0, 10, 0, 4))
    left_again = _mask_candidate(0.8, 1, (0, 10, 0, 5))  # mostly inside "left": rejected
    right = _mask_candidate(0.7, 2, (0, 10, 6, 10))
    chosen = MaskSelector(num_objects=2, max_overlap=0.1)([right, left_again, left])
    assert chosen.mask.sum() == 40 + 40
    assert chosen.score == pytest.approx(0.8)


def test_one_object_returns_the_best_mask_unchanged():
    cands = [_mask_candidate(0.5, 0, (0, 2, 0, 2)), _mask_candidate(0.9, 1, (5, 7, 5, 7))]
    one = MaskSelector(num_objects=1)(cands)
    assert np.array_equal(one.mask, cands[1].mask) and one.score == pytest.approx(0.9)


def test_two_objects_requires_different_points():
    a = _mask_candidate(0.9, 0, (0, 3, 0, 3))
    b = _mask_candidate(0.8, 0, (6, 9, 6, 9))  # disjoint but from the same point
    assert MaskSelector(num_objects=2)([a, b]).mask.sum() == 9
