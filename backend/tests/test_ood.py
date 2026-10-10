import pytest
from conftest import cxr_png

from app.ood import ImageError, check_image, load_image


def test_upright_gray_image_not_flagged_and_colour_flagged():
    assert not check_image(load_image(cxr_png())).ood_flag
    r = check_image(load_image(cxr_png(colour=True)))
    assert r.ood_flag and "colour" in r.reason


def test_rotated_small_and_blank():
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(cxr_png())).rotate(90, expand=True)
    b = io.BytesIO(); img.save(b, "PNG")
    assert check_image(load_image(b.getvalue())).ood_flag
    assert check_image(load_image(cxr_png(60, 80))).ood_flag
    blank = io.BytesIO(); Image.new("L", (300, 300), 100).save(blank, "PNG")
    assert check_image(load_image(blank.getvalue())).ood_flag


def test_bad_inputs_rejected():
    with pytest.raises(ImageError):
        load_image(b"not an image")
    with pytest.raises(ImageError):
        load_image(b"\x00" * 128 + b"DICM" + b"\x00" * 100)
