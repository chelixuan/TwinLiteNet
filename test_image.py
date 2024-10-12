import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
import os
import torch
from model import TwinLite as net
import cv2

from tqdm import tqdm

def Run(model,img):
    img = cv2.resize(img, (1280, 720))
    img_rs=img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0
    img = img.cuda()

    with torch.no_grad():
        img_out = model(img)
    x0 = img_out
    # x0=img_out[0]
    # x1=img_out[1]

    _,da_predict=torch.max(x0, 1)
    # _,ll_predict=torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    # LL = ll_predict.byte().cpu().data.numpy()[0]*255
    # original code 
    # img_rs[DA>100]=[255,0,0]
    # img_rs[LL>100]=[0,255,0]

    # clx ---------------------------------------------------------------------------------------
    color_da = np.array([[(0, 191, 0) for i in range(1280)] for j in range(720)])
    # color_ll = np.array([[(192, 67, 251) for i in range(640)] for j in range(360)])
    
    # clx_original 将结果可视化在原始图片上
    img_rs[DA>100]= img_rs[DA>100] * 0.5 + color_da[DA>100] * 0.5
    # img_rs[LL>100]= img_rs[LL>100] * 0.5 + color_ll[LL>100] * 0.5

    # 二维码数据，尺度为 640*480
    # color_da = np.array([[(0, 191, 0) for i in range(640)] for j in range(480)])
    # color_ll = np.array([[(192, 67, 251) for i in range(640)] for j in range(480)])
    # 不要原始图片信息, 重新初始化一张全黑的 img_rs
    # img_rs = np.array([[(0, 0, 0) for i in range(640)] for j in range(480)])
    # img_rs[DA>100]= color_da[DA>100]
    # # img_rs[LL>100]= color_ll[LL>100]

    # -------------------------------------------------------------------------------------------

    return img_rs


# lyon
MODEL_PATH = '/home/chelx/ckpt/TwinLetNet/lyon2024/single_head/base_bce_500epochs/model_499.pth'
# val
IMAGE_PATH = '/home/chelx/dataset/seg_images/images/val/val_batch_01_202409_lyon/'
RES_SAVE_PATH = MODEL_PATH[:MODEL_PATH.rfind('/')+1] + 'res_val/'

model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
# original code ----------------------------------------------------------------------
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()
# ------------------------------------------------------------------------------------

# clx --------------------------------------------------------------------------------
from collections import OrderedDict
state_dict = torch.load(MODEL_PATH)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k[:8] != 'module.':
        k = 'module.' + k
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
model.eval()
# ------------------------------------------------------------------------------------
image_list=[x for x in os.listdir(IMAGE_PATH) if x[-4:] in ['.jpg', 'png']]

if os.path.exists(RES_SAVE_PATH):
    shutil.rmtree(RES_SAVE_PATH)
os.makedirs(RES_SAVE_PATH, exist_ok=True)
for i, imgName in enumerate(tqdm(image_list)):
    # original code ------------------------------------------------------------------
    # img = cv2.imread(os.path.join(IMAGE_PATH,imgName))
    # img=Run(model,img)
    # cv2.imwrite(os.path.join(RES_SAVE_PATH,imgName),img)
    # --------------------------------------------------------------------------------

    # clx ----------------------------------------------------------------------------
    img = cv2.imread(os.path.join(IMAGE_PATH,imgName))
    orig_h, orig_w, _ = img.shape
    img=Run(model,img)
    img = cv2.resize(img, (orig_w, orig_h))
    cv2.imwrite(os.path.join(RES_SAVE_PATH,imgName),img)
    # --------------------------------------------------------------------------------