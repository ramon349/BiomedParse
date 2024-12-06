from __future__ import annotations
from monai.transforms import MapTransform,Transform
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
from skimage.transform import rescale
from inference_utils.processing_utils import process_intensity_image
import torch 
import numpy as np 
from einops import rearrange
from monai.transforms import Pad,SpatialPad
from collections.abc import Callable, Sequence
from itertools import chain
from math import ceil
from typing import Any
from monai.utils import Method,look_up_option,PytorchPadMode,fall_back_tuple
from monai.transforms.croppad.functional import pad_func
from monai.data.meta_tensor import MetaTensor
import pdb 

#create the manuscript transforms as monai transforms 
class BiomedScale(Transform):
    def __init__(self,site=None,update_meta=True):
        super().__init__()
        self.update_meta =  update_meta
        self.site = site 
    def __call__(self,img):
        img = convert_to_tensor(img, track_meta=get_track_meta()).squeeze(0) 
        img = np.array(img)
        new_img = torch.tensor(process_intensity_image(img,is_CT=True,site=self.site)).unsqueeze(0)
        return new_img
class BiomedScaled(MapTransform):
    def __init__(self, keys=None,site=None,allow_missing_keys = False,update_meta=False) -> None:
        super().__init__(keys, allow_missing_keys) 
        self.converter =  BiomedScale(site=site,update_meta=update_meta) 
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
class Rearranged(MapTransform):
    def __init__(self, keys=None,site=None,allow_missing_keys = False,update_meta=False) -> None:
        super().__init__(keys, allow_missing_keys) 
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] =   rearrange(d[key],"b h w d c -> b c h w d")
        return d

class SliceNorm(Transform): 
    def __init__(self,spatial_axis=-1,update_meta=True) -> None:
        super().__init__()
        spatial_axis= spatial_axis 
        self.update_meta= update_meta
    def __call__(self, data):
        slice_wise_max = data.max(0).values.max(0).values.max(0).values
        slice_wise_min = data.min(0).values.min(0).values.min(0).values
        slice_wise_diff = slice_wise_max - slice_wise_min
        normalized = (data - slice_wise_min)/slice_wise_diff
        return normalized*255
class SliceNormd(MapTransform):
    def __init__(self, keys, allow_missing_keys = False,spatial_axis=-1,update_meta=False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = SliceNorm(spatial_axis=spatial_axis,update_meta=update_meta)
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
class SpatialPadSquare(SpatialPad):
    def __init__(self, spatial_size: Sequence[int] | int | tuple[tuple[int, ...] | int, ...], method: str = Method.SYMMETRIC, mode: str = PytorchPadMode.CONSTANT, lazy: bool = False, **kwargs) -> None:
        super().__init__(spatial_size, method, mode, lazy, **kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> tuple[tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.

        Args:
            spatial_shape: spatial shape of the original image.

        """ 
        h,w,d =  spatial_shape
        if h==w:  #early exit #do not pad 
            return [ [(0,0) for i,sp_i in enumerate(spatial_shape)]]  
        #w_padding 
        width = max((h-w),0)
        w_pad = (int(width//2),int(width-(width//2)))
        #h_padding 
        width = max((w-h),0)
        h_pad = (int(width//2),int(width-(width//2))) 
        ramen_pad = [h_pad,w_pad,(0,0)]
        return tuple([(0,0)]+ramen_pad)
    def __call__(  # type: ignore[override]
        self,
        img: torch.Tensor,
        to_pad: tuple[tuple[int, int]] | None = None,
        mode: str | None = None,
        lazy: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
            to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
                default to `self.to_pad`.
            mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            lazy: a flag to override the lazy behaviour for this call, if set. Defaults to None.
            kwargs: other arguments for the `np.pad` or `torch.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        """
        to_pad_ = self.to_pad if to_pad is None else to_pad
        if to_pad_ is None:
            spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
            to_pad_ = self.compute_pad_width(spatial_shape)
        mode_ = self.mode if mode is None else mode
        kwargs_ = dict(self.kwargs)
        kwargs_.update(kwargs)

        img_t = convert_to_tensor(data=img, track_meta=get_track_meta())
        lazy_ = self.lazy if lazy is None else lazy
        return pad_func(img_t, to_pad_, self.get_transform_info(), mode_, lazy_, **kwargs_)
class SquarePadd(MapTransform):
    def __init__(self,keys,allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.converter = SpatialPadSquare(spatial_size=(719,719,-1),mode='constant',method='symmetric')
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d