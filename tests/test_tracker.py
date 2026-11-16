import numpy as np

from face_swap.tracking import FaceTrackSet, iou_matrix


def box(x, y, s=100):
    return [x, y, x + s, y + s]


def test_iou_matrix():
    m = iou_matrix([box(0, 0)], [box(0, 0), box(50, 0), box(500, 500)])
    assert np.allclose(m[0], [1.0, 1 / 3, 0.0], atol=1e-4)


def test_ids_stable_when_detector_order_flips():
    t = FaceTrackSet()
    a, b = t.update([box(0, 0), box(400, 0)])
    for k in range(1, 20):
        assert t.update([box(400 + k * 3, 0), box(k * 3, 0)]) == [b, a]


def test_ids_follow_first_appearance():
    t = FaceTrackSet()
    assert t.update([box(0, 0)]) == [0]
    assert t.update([box(600, 0), box(2, 0)]) == [1, 0]


def test_short_occlusion_keeps_id_long_one_drops():
    t = FaceTrackSet(max_age=3)
    (a,) = t.update([box(0, 0)])
    for _ in range(3):
        t.update([])
    assert t.update([box(4, 0)]) == [a]
    for _ in range(5):
        t.update([])
    assert t.update([box(4, 0)]) != [a]


def test_appearance_resolves_jumps_and_overlap():
    rng = np.random.default_rng(0)
    ea, eb = rng.normal(size=512), rng.normal(size=512)
    t = FaceTrackSet()
    a, b = t.update([box(0, 0), box(300, 0)], [ea, eb])
    # people swap sides in one jump (a cut): IoU alone would mislabel
    assert t.update([box(300, 0), box(0, 0)], [ea, eb]) == [a, b]
    # heavily overlapping boxes: appearance decides
    assert t.update([box(140, 0), box(160, 0)], [eb, ea]) == [b, a]


def test_tag_faces_sorts_by_track():
    class F:
        def __init__(self, bb):
            self.bbox = np.array(bb, np.float32)

    t = FaceTrackSet()
    t.tag_faces([F(box(0, 0)), F(box(500, 0))])
    out = t.tag_faces([F(box(500, 0)), F(box(0, 0))])
    assert [f.track_id for f in out] == [0, 1] and out[0].bbox[0] == 0
