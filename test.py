import torch
import numpy as np
from models_archs import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import PIL.Image as pil_image
from skimage import io, color
from pathlib import Path
from math import log10,sqrt
from codes.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale = int(input('\nScale: '))
which_version = int(input('\nWhich version (1/2)\n1) ESC-PAN(r, 32, 1)\n2) ESC-PAN(r, 16, 20)\n'))

if which_version == 1:
    C = 32
    d = 1
elif which_version == 2:
        C = 16
        d = 20
else:
    print("Invalid selection :/")
    exit()
model = ESC_PAN(scale,C,d).to(device)
# model = torch.nn.DataParallel(
#         model, device_ids=range(torch.cuda.device_count()))
loaded_checkpoint = torch.load("./saved_models/ESC_PAN_"+str(model.C)+"_"+str(model.d)+"/x"+str(model.scale_factor)+".pth", device)

model.load_state_dict(loaded_checkpoint['state_dict'])
bicubic_flag = False
model.eval()
###################################################################

print("ESC-PAN("+str(scale)+", "+str(model.C)+", "+str(model.d)+")\n")
def test(model,bicubic_flag,scale):
  datasets = ["Set5", "Set14", "BSDS100", "Urban100", "Manga109"]

  for dataset in datasets:
    folder = Path("./data/test/{:s}".format(dataset)).rglob('*.png')
    # if dataset == "DIV2K Valid":
    #   folder = Path("/home/adnanhamida/SUPER_RESOLUTION/TRAINING/DIV2K/120x120_DIV2K_VS/valid").rglob('*.tiff')
    files = [x for x in folder]

    scale = scale
    psnr_val = []
    ssim_val = []
    for file in files:
        img = io.imread(file)

        h = img.shape[0]
        w = img.shape[1]

        if len(img.shape)>2:
            img = color.rgb2ycbcr(img)[:,:,0]

        h = h-np.mod(h,scale)
        w = w-np.mod(w,scale)
        img = img[:h,:w]

        img = pil_image.fromarray(img)
        img_down = img.resize((w//scale, h//scale),resample=pil_image.BICUBIC)
        img_up = img_down.resize((w, h), resample=pil_image.BICUBIC)

        img = np.array(img).astype(np.float32)
        img_down = np.array(img_down).astype(np.float32)
        img_up = np.array(img_up).astype(np.float32)

        im_input = img_down/255.
        if bicubic_flag:
          im_input = img_up/255.
        im_input = (torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

        output = model(im_input.to(device)).clip(0,1)*255
        output = np.array(output[0,0,:,:].detach().cpu()).astype(np.float32)
        #Shaving edges
        img = img[scale:h-scale, scale:w-scale]
        output = output[scale:h-scale, scale:w-scale]

        psnr_val.append(peak_signal_noise_ratio(img, output, data_range=255))
        ssim_val.append(structural_similarity(img,output, gaussian_weights=True, data_range=255))

    print(dataset)
    print("PSNR: {:.2f}".format(np.array(psnr_val).mean()))
    print("SSIM: {:.3f}".format(np.array(ssim_val).mean()))
    print(" ")
test(model=model,scale=model.scale_factor,bicubic_flag=bicubic_flag)
