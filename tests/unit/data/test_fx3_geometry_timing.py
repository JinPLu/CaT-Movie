"""Real failing source metadata and frozen P1a frame dimensions, no media encoding."""
import pytest
import torch
from torchvision.transforms import CenterCrop
from mspc.data.multishot import cover_crop_geometry


@pytest.mark.parametrize('size',[(640,360),(640,272),(854,480)])
def test_declared_crop_offsets_match_actual_torchvision(size):
    g=cover_crop_geometry(video_width=size[0],video_height=size[1],target_width=832,target_height=480)
    rw,rh=g['resize_size']; left,top,right,bottom=g['crop_box_resized']
    # Coordinate ramps verify odd offset rounding, not an independent resize route.
    xs=torch.arange(rw).repeat(rh,1); ys=torch.arange(rh)[:,None].repeat(1,rw)
    actual=CenterCrop((480,832))(torch.stack((xs,ys)))
    assert actual[0,0,0].item()==left
    assert actual[1,0,0].item()==top
    assert actual[0,-1,-1].item()==right-1
    assert actual[1,-1,-1].item()==bottom-1
