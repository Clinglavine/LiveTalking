import torch
import glob
import pickle
import os
from tqdm import tqdm
import cv2

import numpy as np
import copy


from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def load_avatar(avatar_id):
    #self.video_path = '' #video_path
    #self.bbox_shift = opt.bbox_shift
    avatar_path = f"../data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"
    # self.avatar_info = {
    #     "avatar_id":self.avatar_id,
    #     "video_path":self.video_path,
    #     "bbox_shift":self.bbox_shift   
    # }

    input_latent_list_cycle = torch.load(latents_out_path)  #,weights_only=True
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

def load_all_model():
    audio_processor = Audio2Feature(model_path="../models/whisper/tiny.pt")
    vae = VAE(model_path = "../models/sd-vae-ft-mse/")
    unet = UNet(unet_config="../models/musetalk/musetalk.json",
                model_path ="../models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe

def load_model():
    # load model weights
    audio_processor,vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    #vae.vae.share_memory()
    unet.model = unet.model.half()
    #unet.model.share_memory()
    return vae, unet, pe, timesteps, audio_processor


@torch.no_grad()
def warm_up(batch_size,model):
    # 预热函数
    print('warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    #batch_size = 16
    #timesteps = torch.tensor([0], device=unet.device)
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    pred_latents = unet.model(latent_batch,
                              timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)

def __mirror_index(index, coord_list_cycle):
    size = len(coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1  

def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = image
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    face_large[y-y_s:y1-y_s, x-x_s:x1-x_s]=face

    mask_image = cv2.cvtColor(mask_array,cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image/255).astype(np.float32)

    # mask_not = cv2.bitwise_not(mask_array)
    # prospect_tmp = cv2.bitwise_and(face_large, face_large, mask=mask_array)
    # background_img = body[y_s:y_e, x_s:x_e]
    # background_img = cv2.bitwise_and(background_img, background_img, mask=mask_not)
    # body[y_s:y_e, x_s:x_e] = prospect_tmp + background_img

    #print(mask_image.shape)
    #print(cv2.minMaxLoc(mask_image))

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(face_large,body[y_s:y_e, x_s:x_e],mask_image,1-mask_image)

    #body.paste(face_large, crop_box[:2], mask_image)
    return body


def update_month(avatar_id):
    model = load_model()
    warm_up(batch_size=16, model=model)   
    vae, unet, pe, timesteps, audio_processor = model
    frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle = load_avatar(avatar_id)
    output_dir = os.path.join(f'../data/avatars/{avatar_id}', 'full_imgs1')
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in tqdm(range(len(frame_list_cycle)), desc="Processing frames"):
        silent_audio_feature = np.zeros((1, 50, 384), dtype=np.uint8)  # 假设静音特征为全零
        audio_feature_batch = pe(
            torch.from_numpy(silent_audio_feature).to(
                device=unet.device,
                dtype=unet.model.dtype
                )
            )
        latent_batch = input_latent_list_cycle[__mirror_index(idx, coord_list_cycle)].unsqueeze(0)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        # 调整形状移除多余的维度
        latent_batch = latent_batch.squeeze(1)
        # 推理生成静音对应的帧
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        
        recon_frame = vae.decode_latents(pred_latents)[0]
        if isinstance(recon_frame, torch.Tensor):  # 如果是 torch.Tensor 类型
            recon_frame = recon_frame.cpu().numpy()  # 转换为 numpy.ndarray

        # 调整静音推理帧与背景帧融合
        ori_frame = copy.deepcopy(frame_list_cycle[idx])
        bbox = coord_list_cycle[idx]
        x1, y1, x2, y2 = bbox
        target_width = x2 - x1
        target_height = y2 - y1
        try:
            recon_frame_resized = cv2.resize(recon_frame, (target_width, target_height))
        except:
            continue
        combine_frame = get_image_blending(ori_frame, recon_frame_resized, bbox, mask_list_cycle[idx], mask_coords_list_cycle[idx])
        
        output_path = os.path.join(output_dir, f"{str(idx).zfill(8)}.png")
        cv2.imwrite(output_path, combine_frame)
    print("finsh")

update_month(avatar_id="avator_9")
