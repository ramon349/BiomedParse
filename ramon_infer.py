
import sys 
sys.path.append("../BiomedParse/")
import monai 
from inference_utils.processing_utils import process_intensity_image
from inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import process_intensity_image
from PIL import Image 
from skimage.transform import rescale
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
import pickle as pkl 
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    SaveImaged
)
import matplotlib.pyplot as plt
from monai.data import NibabelReader,Dataset
from monai.transforms import MapTransform,Transform
from monai.utils import convert_to_tensor
from monai.data.meta_obj import get_track_meta
import numpy as np 
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from hashlib import sha224
import pandas as pd 
import argparse 
from torch.nn import functional as F
from tqdm import tqdm 
import os
def subject_formater(meta_in, ignore):
    pid = meta_in["filename_or_obj"]
    out_form = sha224(pid.encode("utf-8")).hexdigest()
    return {"subject": f"{out_form}", "idx": "0"}
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

def load_model():
    """ Hardcoded the model loading logic. TODO: Tehre is an empty_weight error for one param. should investigate 
    """
    conf_files = "./configs/biomedparse_inference.yaml"
    opt = load_opt_from_config_files([conf_files])
    opt = init_distributed(opt)
    model_file = "../model_pretrained/biomedparse_v1.pt"

    model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    return model 
def pred_wrap(model,vol_slice): 
    """ Custom forward method. Converts to RGB PIL from tensor then runs 
    inference and back to tensor. TODO: Make prompts customizable 
    """
    data = rearrange(vol_slice,"b c h w l -> b h w c l").squeeze(-1).squeeze(0)
    arr = np.array(data).astype(np.uint8)
    #arr =np.reshape(arr,(1024,1024,3))
    data= Image.fromarray(arr)
    out = interactive_infer_image(model,data,['Segment all kidneys'])
    tens = torch.tensor(out).unsqueeze(-1).unsqueeze(0).to(torch.float)
    return  tens 
def check_if_exists(my_tensors,output_dir): 
    #little helper i'm keeping around just incase
    output_stuff = subject_formater(my_tensors.meta,None)['subject']
    output_file = os.path.join(output_dir,output_stuff,f"{output_stuff}_seg_0.nii.gz")
    return  os.path.isfile(output_file)
def get_args(): 
    parser =  argparse.ArgumentParser(description="") 
    parser.add_argument("--data_path",required=True,type=str,help='Path To pickle file') 
    parser.add_argument("--output_dir",required=True,type=str,help="Path To store output_segs")
    parser.add_argument("--inference_map_path",required=True,type=str) 
    parser.add_argument("--name",required=True,type=str)
    my_args = parser.parse_args() 
    arg_d = vars(my_args)
    return arg_d

def main(): 
    #Transforms needed to privde images in the format expected by the model
    transforms = Compose(
        [
        LoadImaged(keys=['image','label'],reader=NibabelReader,image_only=False), #need the metadata for saving
        EnsureChannelFirstd(keys=['image','label']),
        Orientationd(keys=['image','label'],axcodes='RAS'), # i prefer using ras 
        BiomedScaled(keys=['image'],site='abdomen'), # scale intensities as done in the model 
        Rearranged(keys=['image'])
        ]
    )
    conf = get_args()
    dset_name = conf['name'] 
    pkl_path = conf['data_path'] 
    out_dir = conf['output_dir']
    inference_map_path = conf['inference_map_path'] 
    print(f"Using Path {pkl_path}") 
    print(f"Images stored: {out_dir}") 
    print(f"Segmentation csv will be in dir {inference_map_path}")
    model = load_model().to('cuda')
    # i have my datasets in a pkl file (tra,val,ts)  
    #ts = [{'image':..,'label':...}]
    #loads test set only 
    with open(pkl_path,'rb') as f:
        ts =  pkl.load(f)[-1]
    tr_ds = Dataset(ts,transform=transforms)
    # Per docs we can only have a batchsize of 1
    dl = DataLoader(tr_ds,batch_size=1,num_workers=4,persistent_workers=True)
    all_paths = list() 
    preds = list() 
    post_transforms = Compose(
            [
                SaveImaged(
                    keys=["preds"],
                    output_dir=out_dir,
                    meta_keys=['image_meta_dict'],
                    output_postfix="seg",
                    resample=False,
                    data_root_dir="",
                    savepath_in_metadict=True,
                    output_ext='nii.gz',
                    output_name_formatter=subject_formater,
                   meta_key_postfix=""
                )
            ]
        )
    for data_sample in tqdm(dl,total=len(dl)): 
        #Use Sliding window inference to infer across Z dimension of slices 
        slide_preds = sliding_window_inference(data_sample['image'][0].to(torch.float),roi_size=(1024,1024,1),sw_batch_size=1,predictor=lambda x: pred_wrap(model,x),sw_device='cpu',device='cpu',buffer_dim=-1,progress=False)
        #make a metatensor to preserve some important info
        data_sample['preds']= monai.data.MetaTensor(slide_preds,affine=data_sample['label'].meta['affine'],meta=data_sample['label'].meta)
        #Reisze output image from 1024,1024,-1 to original image size 
        orig_size= data_sample['label'].shape[-3:] 
        data_sample['preds'] =  F.interpolate(data_sample['preds'],size=orig_size,mode='nearest')
        #save the image
        other = post_transforms(data_sample) 
        # get  file paths for later
        orig_path = other['preds'].meta['filename_or_obj']
        saved_path = other['preds'].meta['saved_to']
        all_paths.append(orig_path)
        preds.append(saved_path)

    #make and save a csv in the desired path
    all_preds = pd.DataFrame({'orig':all_paths,'pred':preds})
    all_preds['name']= dset_name
    os.makedirs(inference_map_path,exist_ok=True)
    f_path = os.path.join(inference_map_path,f"{dset_name}_inference.csv")
    all_preds.to_csv(f_path,index=False)

if __name__=='__main__':
    main()