import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import os
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap
current_dir = os.path.dirname(os.path.abspath(__file__))
from mini_live.render import create_render_model
import pickle
import cv2
import time
import numpy as np
import glob
import random
import os
import sys
import torch
from talkingface.model_utils import LoadAudioModel, Audio2bs
from talkingface.data.few_shot_dataset import get_image

def run(video_path, pkl_path, wav_path, output_video_path):
    Audio2FeatureModel = LoadAudioModel(r'checkpoint/lstm/lstm_model_epoch_325.pkl')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mouth_fusion_mask = cv2.imread("mini_live/mouth_fusion_mask.png").astype(float) / 255.

    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("checkpoint/FreeFace_3ref/epoch_60.pth")

    start_time = time.time()
    standard_size = 320
    crop_rotio = [0.5, 0.5, 0.5, 0.5]
    out_w = int(standard_size * (crop_rotio[0] + crop_rotio[1]))
    out_h = int(standard_size * (crop_rotio[2] + crop_rotio[3]))
    out_size = (out_w, out_h)
    renderModel_gl = create_render_model((out_w, out_h), floor=20)

    from mini_live.obj.obj_utils import generateWrapModel
    from talkingface.utils import crop_mouth, main_keypoints_index
    wrapModel, wrapModel_face = generateWrapModel()

    with open(pkl_path, "rb") as f:
        images_info = pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    vid_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    list_source_crop_rect = []
    list_video_img = []
    list_standard_img = []
    list_standard_v = []
    list_standard_vt = []
    for frame_index in range(min(vid_frame_count, len(images_info))):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        source_pts = images_info[frame_index]
        source_crop_rect = crop_mouth(source_pts[main_keypoints_index], vid_width, vid_height)

        standard_img = get_image(frame, source_crop_rect, input_type="image", resize=standard_size)
        standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize=standard_size)
        standard_vt = standard_v[:, :2] / standard_size

        list_video_img.append(frame)
        list_source_crop_rect.append(source_crop_rect)
        list_standard_img.append(standard_img)
        list_standard_v.append(standard_v)
        list_standard_vt.append(standard_vt)
    cap.release()

    renderModel_mini.reset_charactor(list_standard_img, np.array(list_standard_v)[:, main_keypoints_index])

    # 求平均人脸
    from talkingface.run_utils import calc_face_mat
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v),
                                                               renderModel_gl.face_pts_mean)
    face_wrap_entity = wrapModel.copy()
    # 正规点
    face_wrap_entity[:len(index_wrap), :3] = face_pts_mean_personal_primer[index_wrap, :3]
    # 边缘点
    vert_mid = face_wrap_entity[:, :3][index_edge_wrap[:4] + index_edge_wrap[-4:]].mean(axis=0)
    for index_, jj in enumerate(index_edge_wrap):
        face_wrap_entity[len(index_wrap) + index_, :3] = face_wrap_entity[jj, :3] + (
                face_wrap_entity[jj, :3] - vert_mid) * (0.3 - abs(10 - index_) / 40)
    # 牙齿点
    from talkingface.utils import INDEX_LIPS, main_keypoints_index, INDEX_LIPS_UPPER, INDEX_LIPS_LOWER
    # 上嘴唇中点
    mid_upper_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS_UPPER], axis=0)
    mid_upper_teeth = np.mean(face_wrap_entity[-36:-18, :3], axis=0)
    tmp = mid_upper_teeth - mid_upper_mouth
    face_wrap_entity[-36:-18, :2] = face_wrap_entity[-36:-18, :2] - tmp[:2]
    # 下嘴唇中点
    mid_lower_mouth = np.mean(face_pts_mean_personal_primer[main_keypoints_index][INDEX_LIPS_LOWER], axis=0)
    mid_lower_teeth = np.mean(face_wrap_entity[-18:, :3], axis=0)
    tmp = mid_lower_teeth - mid_lower_mouth
    print(tmp)
    face_wrap_entity[-18:, :2] = face_wrap_entity[-18:, :2] - tmp[:2]

    renderModel_gl.GenVBO(face_wrap_entity)

    bs_array = Audio2bs(wav_path, Audio2FeatureModel)[5:] * 0.5
    import uuid
    task_id = str(uuid.uuid1())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "{}.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))

    for frame_index in range(len(mat_list)):
        if frame_index >= len(bs_array):
            continue
        bs = np.zeros([12], dtype=np.float32)
        bs[:6] = bs_array[frame_index, :6]
        # bs[2] = frame_index* 5

        verts_frame_buffer = np.array(list_standard_vt)[frame_index, index_wrap, :2].copy() * 2 - 1

        rgba = renderModel_gl.render2cv(verts_frame_buffer, out_size=out_size, mat_world=mat_list[frame_index].T,
                                        bs_array=bs)
        rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

        face_mask = (rgba[:, :, 2] > 200) | ((rgba[:, :, 2] < 100))
        face_mask = face_mask.astype(np.uint8)
        face_mask = np.concatenate(
            [face_mask[:, :, np.newaxis], face_mask[:, :, np.newaxis], face_mask[:, :, np.newaxis]], axis=2)

        wrap_rgb = rgba[:, :, :2]
        # print(rgba[:,:,0].mean(),rgba[:,:,1].mean(),rgba[:,:,2].mean())
        deformation = (wrap_rgb.astype(float) / 255. * 2 - 1) * 320 * 0.5

        # 假设deformation的前两通道分别代表X和Y方向的位移
        x_displacement = deformation[:, :, 0]
        y_displacement = deformation[:, :, 1]

        # 创建网格坐标
        x, y = np.meshgrid(np.arange(320), np.arange(320))

        # 计算新的映射
        map_x = (x - x_displacement).astype(np.float32)
        map_y = (y - y_displacement).astype(np.float32)

        # # 读取待变形的图像
        # img = cv2.imread(img_filenames[frame_index])

        img0 = list_standard_img[frame_index]
        # 使用cv2.remap进行图像变形
        warped_img = cv2.remap(img0, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
        warped_img = warped_img * (1 - face_mask) + rgba * face_mask

        warped_img2 = img0.copy()
        pad = 5
        warped_img2[pad:-pad, pad:-pad] = warped_img[pad:-pad, pad:-pad]
        fake_face = cv2.resize(warped_img2, (160, 160))
        fake_mouth = fake_face[52:-52, 44:-44]
        # print(rgb_mouth.shape)
        # cv2.imshow('scene', rgb_mouth)
        # cv2.waitKey(-1)

        fake_mouth2 = renderModel_mini.interface(fake_mouth)

        fake_mouth2 = fake_mouth2 * mouth_fusion_mask + fake_mouth * (1 - mouth_fusion_mask)
        fake_face[52:-52, 44:-44] = fake_mouth2

        x_min, y_min, x_max, y_max = list_source_crop_rect[frame_index]

        img_face = cv2.resize(fake_face, (x_max - x_min, y_max - y_min))
        img_bg = list_video_img[frame_index]
        img_bg[y_min:y_max, x_min:x_max] = img_face
        # cv2.imshow('scene', img_bg[:,:,::-1])
        # cv2.waitKey(10)
        # print(time.time())

        videoWriter.write(img_bg[:, :, ::-1])
    videoWriter.release()

    os.system(
        "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p {}".format(save_path, wav_path, output_video_path))
    os.remove(save_path)

    cv2.destroyAllWindows()

def main():
    # 检查命令行参数的数量
    if len(sys.argv) < 4:
        print("Usage: python demo.py <video_path> <audio_path> <output_video_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_path = sys.argv[1]
    print(f"Video path is set to: {video_path}")
    wav_path = sys.argv[2]
    print(f"Audio path is set to: {wav_path}")
    output_video_name = sys.argv[3]
    print(f"output video name is set to: {output_video_name}")
    try:
        model_name = sys.argv[4]
        print(f"model_name: {model_name}")
    except Exception:
        model_name = "render.pth"

    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)

    run(video_path, pkl_path, wav_path, output_video_name)

# 示例使用
if __name__ == "__main__":
    main()



