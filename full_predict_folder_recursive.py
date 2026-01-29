from pathlib import Path
import os
import shutil
import warnings
import time

import numpy as np
import predict_one_image as pred_im
import predict_one_image_av as pred_im_av
from utils import paired_transforms_tv04 as p_tr
from PIL import Image, ImageFont, ImageDraw

from skimage.io import imsave, imread
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb
import torch
from models.get_model import get_arch
from utils.model_saving_loading import load_model
import random

from skimage.transform import resize
from skimage.measure import regionprops

def main():
    rand_count = 10

    out_dir = '/mnt/d/Retinal output'

    if True:
        # '/mnt/d/Retinal data/BHC retinal image_all/BHC retinal image_all',
                        
        dataset_dirs = ['/mnt/d/Retinal data/BHC retinal image_all/BHC retinal image_all',
                        '/mnt/d/Retinal data/isave retinal/NW400 FP',
                        '/mnt/d/Retinal data/isave retinal/TRC-50DX FP',
                        '/mnt/d/Retinal data/isave retinal/Triton FP',
                        '/mnt/d/Retinal data/SEED retinal image/Imaging']
        dataset_dirs = ['/mnt/d/Retinal data/isave retinal/NW400 FP',
                        '/mnt/d/Retinal data/isave retinal/TRC-50DX FP',
                        '/mnt/d/Retinal data/isave retinal/Triton FP']

        files = []
        for dd in dataset_dirs:
            subd = [p for p in Path(dd).iterdir() if p.is_dir() and not os.path.isdir(str(p).replace('/mnt/d/Retinal data', out_dir))]

            if len(subd) > rand_count:
                subd = random.sample(subd, rand_count)
            #print(subd)
            #print('')

            for ss in subd:
                files.extend([file_path for file_path in Path(ss).rglob("*.[jJ][pP][gG]") if not file_path.name.startswith('.')])
                files.extend([file_path for file_path in Path(ss).rglob("*.[tT][iI][fF]") if not file_path.name.startswith('.')])
                files.extend([file_path for file_path in Path(ss).rglob("*.[pP][nN][gG]") if not file_path.name.startswith('.')])
    else:
        #base_dir = '/mnt/d/Retinal data/BHC retinal image_all/BHC retinal image_all/H01-001'
        base_dir = '/mnt/d/Retinal data/SEED retinal image/Imaging/SEED008_C49/SEED008_C49_1'
        p = Path(base_dir)

        files = [file_path for file_path in p.rglob("*.[jJ][pP][gG]") if not file_path.name.startswith('.')]


    print(len(files))

    for f in files:
        src_folder = os.path.split(f)[0]
        fn = os.path.split(f)[1]
        print(src_folder)
        tar_folder = src_folder.replace('/mnt/d/Retinal data', out_dir)

        print(tar_folder)

        Path(tar_folder).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src_folder+'/'+fn, tar_folder+'/'+fn) 
        
        tfn = tar_folder+'/'+fn
        print('')
        print(fn)
        print('')
        try:
            _, p_drv = run_predict(tfn, 'wnet', 'experiments/wnet_drive/', 512, 'DRIVE', bin_thresh=0.4196)
            _, p_hrf = run_predict(tfn, 'wnet', 'experiments/wnet_hrf_1024/', 1024, 'HRF', bin_thresh=0.3725)
            p_drv_av, _ = run_predict(tfn, 'big_wnet', 'experiments/big_wnet_drive_av/', 512, 'DRIVE_AV')
            p_hrf_av, _ = run_predict(tfn, 'big_wnet', 'experiments/big_wnet_hrf_av_1024/', 1024, 'HRF_AV')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue

        im = Image.open(tfn)
        mask = pred_im.get_fov(im)
        mask = np.array(mask).astype(int)
        minr, minc, maxr, maxc = regionprops(mask)[0].bbox
        print([minr,maxr,minc,maxc])

        im = np.asarray(im)
        im = im[minr:maxr, minc:maxc, :]
        p_drv_av = p_drv_av[minr:maxr, minc:maxc, :]
        p_hrf_av = p_hrf_av[minr:maxr, minc:maxc, :]
        p_drv = p_drv[minr:maxr, minc:maxc]
        p_hrf = p_hrf[minr:maxr, minc:maxc]

        merge_im = np.zeros((512*2,512*3,3),dtype=np.uint8)
        merge_im[0:512,0:512,:] = img_as_ubyte(resize(im,(512,512)))
        merge_im[0:512,512:1024,:] = img_as_ubyte(resize(p_hrf_av,(512,512)))
        merge_im[512:1024,512:1024,:] = img_as_ubyte(resize(p_drv_av,(512,512)))
        merge_im[0:512,1024:1536,:] = img_as_ubyte(resize(gray2rgb(p_hrf),(512,512)))
        merge_im[512:1024,1024:1536,:] = img_as_ubyte(resize(gray2rgb(p_drv),(512,512)))

        merge_im = Image.fromarray(merge_im)

        font = ImageFont.truetype('/usr/share/fonts//truetype/ubuntu/UbuntuMono-B.ttf', 10)
        draw = ImageDraw.Draw(merge_im)
        draw.text((10,712),src_folder.replace('/mnt/d/Retinal data', '...'),(255,255,255),font=font)
        font = ImageFont.truetype('/usr/share/fonts//truetype/ubuntu/UbuntuMono-B.ttf', 14)
        draw.text((10,742),fn,(255,255,255),font=font)

        font = ImageFont.truetype('/usr/share/fonts//truetype/ubuntu/UbuntuMono-B.ttf', 20)
        draw.text((512,10),'HRF A/V',(255,255,255),font=font)
        draw.text((512,522),'DRIVE A/V',(255,255,255),font=font)
        draw.text((1024,10),'HRF (basic)',(255,255,255),font=font)
        draw.text((1024,522),'DRIVE (basic)',(255,255,255),font=font)
        merge_im.save(tar_folder + '/lwnet_out/' + fn.rsplit('.',1)[-2]+'_merge.png')



def run_predict(im_path, model_name, model_path, im_size, postfix, bin_thresh=None):
    device = torch.device("cuda")
    mask_path = None

    if model_name == 'wnet':
        tta = 'from_preds'
        n_classes = 1
    else:
        tta = 'from_probs'
        n_classes = 4

    im_loc = os.path.dirname(im_path)
    im_name = im_path.rsplit('/', 1)[-1]

    result_path = im_loc + '/lwnet_out'
    os.makedirs(result_path, exist_ok=True)
    im_path_out = os.path.join(result_path, im_name.rsplit('.', 1)[-2]+'_'+postfix+'_seg.png')
    im_path_out_bin = os.path.join(result_path, im_name.rsplit('.', 1)[-2]+'_'+postfix+'_bin_seg.png')
    im_path_out_mask = os.path.join(result_path, im_name.rsplit('.', 1)[-2]+'_'+postfix+'__fovmask.png')

    tg_size = (im_size, im_size)

    print('* Segmenting image ' + im_path)
    img = Image.open(im_path)
    mask = pred_im.get_fov(img)
    mask = np.array(mask).astype(bool)
    #imsave(im_path_out_mask, img_as_ubyte(mask))

    img, coords_crop = pred_im.crop_to_fov(img, mask)
    original_sz = img.size[1], img.size[0]  # in numpy convention

    rsz = p_tr.Resize(tg_size)
    tnsr = p_tr.ToTensor()
    tr = p_tr.Compose([rsz, tnsr])
    im_tens = tr(img)  # only transform image

    print('* Instantiating model  = ' + str(model_name))

    model = get_arch(model_name, n_classes=n_classes).to(device)
    model.mode='eval'

    print('* Loading trained weights from ' + model_path)
    model, stats = load_model(model, model_path, device)
    model.eval()

    print('* Saving prediction to ' + im_path_out)
    start_time = time.perf_counter()


    if model_name == 'wnet':
        full_pred, full_pred_bin = pred_im.create_pred(model, im_tens, mask, coords_crop, original_sz, bin_thresh=bin_thresh, tta=tta)
    else:
        full_pred, full_pred_bin = pred_im_av.create_pred(model, im_tens, mask, coords_crop, original_sz, tta=tta)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(im_path_out, img_as_ubyte(full_pred))
        imsave(im_path_out_bin, img_as_ubyte(full_pred_bin))
    print('Done, time spent = {:.3f} secs'.format(time.perf_counter() - start_time))

    return img_as_ubyte(full_pred), img_as_ubyte(full_pred_bin)

if __name__ == '__main__':
    main()