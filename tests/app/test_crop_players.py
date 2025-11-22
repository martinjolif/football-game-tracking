from app.crop_players import yolo_to_bbox

def test_yolo_to_bbox():
    assert yolo_to_bbox(xc=1/2, yc=1/2, w=1/2, h=1/2, img_w=640, img_h=380, padding=0) == (160, 95, 480, 285)
