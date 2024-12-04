import os.path
import torch
import os
import numpy as np
import time
from talkingface.utils import generate_face_mask, INDEX_LIPS_UPPER, INDEX_LIPS_LOWER, crop_mouth, main_keypoints_index

from talkingface.data.few_shot_dataset import select_ref_index
device = "cuda" if torch.cuda.is_available() else "cpu"
import pickle
import cv2

def draw_mouth_maps(keypoints, size=(256, 256), im_edges = None):
    w, h = size
    # edge map for face region from keypoints
    if im_edges is None:
        im_edges = np.zeros((h, w, 3), np.uint8)  # edge map for all edges
    pts = keypoints[INDEX_LIPS_UPPER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(im_edges, [pts], color=(255, 0, 0))
    pts = keypoints[INDEX_LIPS_LOWER]
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(im_edges, [pts], color=(127, 0, 0))
    return im_edges

class RenderModel_Mini:
    def __init__(self):
        self.__net = None

    def loadModel(self, ckpt_path):
        from talkingface.models.DINet_mini import DINet_mini as DINet
        n_ref = 3
        source_channel = 3
        ref_channel = n_ref * 6
        self.__net = DINet(source_channel, ref_channel).cuda()
        checkpoint = torch.load(ckpt_path)
        net_g_static = checkpoint['state_dict']['net_g']
        self.__net.load_state_dict(net_g_static)
        self.__net.eval()


    def reset_charactor(self, img_list, driven_keypoints):
        ref_img_index_list = select_ref_index(driven_keypoints, n_ref=3, ratio=0.33)  # 从当前视频选n_ref个图片
        ref_img_list = []
        for i in ref_img_index_list:
            ref_face_edge = draw_mouth_maps(driven_keypoints[i], size=(320, 320))
            ref_img = img_list[i]
            ref_face_edge = cv2.resize(ref_face_edge, (160, 160))
            ref_img = cv2.resize(ref_img, (160, 160))
            ref_img = np.concatenate([ref_img[53:-53, 44:-44], ref_face_edge[53:-53, 44:-44]], axis=2)

            ref_img_list.append(ref_img)
        self.ref_img = np.concatenate(ref_img_list, axis=2)

        ref_tensor = torch.from_numpy(self.ref_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
        self.__net.ref_input(ref_tensor)


    def interface(self, mouth_frame):
        # tensor
        source_tensor = torch.from_numpy(mouth_frame / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
        fake_out = self.__net.interface(source_tensor)
        image_numpy = fake_out.detach().squeeze(0).cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)
        return image_numpy

    def save(self, path):
        torch.save(self.__net.state_dict(), path)