from __future__ import annotations
from monai.transforms import MapTransform,Transform
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
from skimage.transform import rescale
from inference_utils.processing_utils import process_intensity_image
import torch 
import numpy as np 
from einops import rearrange
from monai.transforms import Pad,SpatialPad,SpatialPadd,Padd
from collections.abc import Callable, Sequence
from itertools import chain
from math import ceil
from typing import Any
from monai.utils import Method,look_up_option,PytorchPadMode,fall_back_tuple
from monai.transforms.croppad.functional import pad_func
from monai.data.meta_tensor import MetaTensor
import pdb 
from monai.config import IndexSelection, KeysCollection, SequenceStr

from monai.transforms.utils_pytorch_numpy_unification import allclose, concatenate, stack
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.transforms.utils import create_rotate, create_translate, resolves_modes, scale_affine
from monai.transforms import MapTransform
from monai.transforms.inverse import InvertibleTransform,TraceableTransform
from skimage.transform import resize
from monai.utils.enums import GridPatchSort, PatchKeys, TraceKeys, TransformBackends
from monai.utils import convert_to_dst_type,convert_to_numpy,convert_to_tensor
from monai.transforms.spatial.functional import _maybe_new_metatensor
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
    def __init__(self,spatial_axis=-1,update_meta=True,a_min=None,a_max=None) -> None:
        super().__init__()
        spatial_axis= spatial_axis 
        self.update_meta= update_meta
        self.a_min= a_min
        self.a_max = a_max
    def __call__(self, data):
        for i in range(data.shape[-1]): 
            slice = data[:,:,:,i]  
            slice= torch.clip(slice,min=self.a_min,max=self.a_max)
            slice_wise_max = slice.max() #min(slice.max(),self.a_max)
            slice_wise_min =  slice.min() #max(slice.min(),self.a_min)
            slice_wise_diff = slice_wise_max - slice_wise_min
            data[:,:,:,i] = ((slice - slice_wise_min)/slice_wise_diff)*255.0
        return data
class SliceNormd(MapTransform):
    def __init__(self, keys, allow_missing_keys = False,spatial_axis=-1,update_meta=False,a_min=None,a_max=None) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = SliceNorm(spatial_axis=spatial_axis,update_meta=update_meta,a_min=a_min,a_max=a_max)
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
class SpatialPadSquare(SpatialPad):
    def __init__(self, spatial_size: Sequence[int] | int | tuple[tuple[int, ...] | int, ...], method: str = Method.SYMMETRIC, mode: str = PytorchPadMode.CONSTANT, lazy: bool = False, **kwargs) -> None:
        super().__init__(spatial_size, method, 'constant', lazy, value=0.0,**kwargs)

    def compute_pad_width(self, spatial_shape: Sequence[int]) -> tuple[tuple[int, int]]:
        """
        dynamically compute the pad width according to the spatial shape.

        Args:
            spatial_shape: spatial shape of the original image.

        """ 
        h,w,d =  spatial_shape
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

class SquarePadd(Padd):
    def __init__(self, keys: KeysCollection, spatial_size: Sequence[int] | int, method: str = Method.SYMMETRIC, mode: SequenceStr = PytorchPadMode.CONSTANT, allow_missing_keys: bool = False, lazy: bool = False, **kwargs) -> None:
        padder = SpatialPadSquare(spatial_size, method, lazy=lazy, **kwargs)
        Padd.__init__(self, keys, padder=padder, mode=mode, allow_missing_keys=allow_missing_keys, lazy=lazy)

class SkResized(MapTransform,InvertibleTransform):
    """
    Resize the input image using skimage's resize function.

    Args:
        keys: keys of the corresponding items to be transformed.
        output_shape: the output shape (spatial dimensions only)
        order: The order of the spline interpolation. See skimage.transform.resize for details.
        mode: Points outside the boundaries of the input are filled according to the given mode. See skimage.transform.resize for details.
        cval: Used in conjunction with mode 'constant', the value outside the image boundaries.
        clip: Whether to clip the output to the range of values of the input image. This prevents the output from having values outside [data.min(), data.max()].
        preserve_range: Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.
        anti_aliasing: Whether to apply a Gaussian filter to smooth the image prior to down-scaling. It is crucial to filter when down-sampling the image to avoid aliasing artifacts.
    """

    def __init__(self, keys, spatial_dim=(10,10), order=3, mode='constant', cval=0, clip=True, preserve_range=True, anti_aliasing=True):
        super().__init__(keys)
        self.spatial_dim =spatial_dim 
        self.order = order
        self.order = {'image':3,'label':3}
        self.mode = mode
        self.cval = cval
        self.clip = clip
        self.preserve_range = preserve_range
        self.anti_aliasing = anti_aliasing
        self.converter = SkResize(spatial_dim=self.spatial_dim,order=self.order,mode=self.mode,cval=self.cval,clip=self.clip,preserve_range=self.preserve_range,anti_aliasing=self.anti_aliasing)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key]= self.converter(d[key],order=self.order[key])
        return d
    def inverse(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter.inverse(d[key],order=self.order[key])
        return d

class SkResize(InvertibleTransform):
    def __init__(self, spatial_dim=(10,10), order=3, mode='constant', cval=0, clip=True, preserve_range=True, anti_aliasing=True):
        super().__init__()
        self.spatial_dim =spatial_dim 
        self.order = {'image':3,'label':3}
        self.mode = mode
        self.cval = cval
        self.clip = clip
        self.preserve_range = preserve_range
        self.anti_aliasing = anti_aliasing,
    def __call__(self, data: Any,order):
        output_shape = (self.spatial_dim[0],self.spatial_dim[1],data.shape[-1])
        new_data = self._resize_wrap(data, output_shape=(1024,1024), order=order, mode=self.mode, cval=self.cval, clip=self.clip, preserve_range=self.preserve_range, anti_aliasing=self.anti_aliasing)
        #self.push_transform(new_data)
        return new_data 
    def _resize_wrap(self,vol:MetaTensor ,output_shape=None,order=None,mode=None,cval=None,preserve_range=None,anti_aliasing=None,clip=None):
        img = convert_to_tensor(vol,track_meta=get_track_meta())
        orig_size = img.peek_pending_shape() if isinstance(img,MetaTensor) else img.shape[1:] 
        true_out_shape = (1024,1024,vol.shape[-1])
        meta_info = TraceableTransform.track_transform_meta(
            img,
            sp_size=(1024,1024),
            affine=scale_affine(orig_size, true_out_shape),
            extra_info={},
            orig_size=orig_size,
            transform_info=self.get_transform_info(),
            lazy=False,
        )
        out = _maybe_new_metatensor(img) 
        img_ = convert_to_numpy(img.squeeze(0))
        resized = self.my_resize(img_,output_shape=true_out_shape,order=order)
        out, *_= convert_to_dst_type(resized,out,dtype=torch.float32)
        out = out.unsqueeze(0) 
        return out.copy_meta_from(meta_info) if isinstance(out,MetaTensor) else out 

    def inverse(self, data: torch.Tensor,order) -> torch.Tensor:
        transform = self.pop_transform(data,check=True)
        return self.inverse_transform(data, transform,order)
    def inverse_transform(self, data: torch.Tensor, transform,order) -> torch.Tensor:
        orig_size = transform[TraceKeys.ORIG_SIZE]
        out = _maybe_new_metatensor(data)
        if data.shape[0] >1: 
            img_=convert_to_numpy(data[0].squeeze(0))
        else: 
            img_=convert_to_numpy(data.squeeze(0))
        resized = self.my_resize(img_,output_shape=orig_size,order=order)
        out,*_ = convert_to_dst_type(resized,out,dtype=torch.float32)
        out = out.unsqueeze(0)
        out.copy_meta_from(data)
        return  out  
    def my_resize(self,vol,output_shape,order):
            new_vol = np.zeros(output_shape)
            for i in range(vol.shape[-1]): 
                new_vol[:,:,i]  = resize(vol[:,:,i],output_shape=(output_shape[0],output_shape[1]),order=order,mode=self.mode,cval=self.cval,clip=self.clip,preserve_range=self.preserve_range)
            return new_vol
