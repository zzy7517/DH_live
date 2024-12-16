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
import threading
import queue


# 假设你有一个函数可以持续提供音频数据
# 例如，从麦克风或网络流中获取
def get_audio_stream():
    # 这里需要替换成你实际的音频流获取代码
    # 示例：返回一个随机的音频数据
    time.sleep(0.1)  # 模拟音频流的间隔
    return np.random.rand(16000)  # 假设每次返回16000个采样点的音频数据


def process_audio(audio_queue, bs_queue, Audio2FeatureModel):
    """
    音频处理线程，持续接收音频数据并转换为特征向量
    """
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:  # 接收到结束信号
            break
        bs = Audio2bs(audio_data, Audio2FeatureModel)[5:] * 0.5
        bs_queue.put(bs)


def process_video(video_path, pkl_path, output_video_path, audio_queue, bs_queue):
    """
    视频处理线程，持续接收音频特征向量并生成视频帧
    """
    Audio2FeatureModel = LoadAudioModel(r'checkpoint/lstm/lstm_model_epoch_325.pkl')

    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("checkpoint/DINet_mini/epoch_40.pth")

    standard_size = 256
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

    images_info = np.concatenate([images_info, images_info[::-1]], axis=0)

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
    from talkingface.run_utils import calc_face_mat
    mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v), renderModel_gl.face_pts_mean)
    from mini_live.obj.wrap_utils import newWrapModel
    face_wrap_entity = newWrapModel(wrapModel, face_pts_mean_personal_primer)

    renderModel_gl.GenVBO(face_wrap_entity)

    import uuid
    task_id = str(uuid.uuid1())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "{}.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))

    frame_index = 0
    while True:
        try:
            bs = bs_queue.get(timeout=1)  # 从队列中获取音频特征向量，设置超时时间
        except queue.Empty:
            print("音频处理结束，视频处理也结束")
            break  # 如果音频处理结束，则视频处理也结束

        if frame_index >= len(mat_list):
            frame_index = 0  # 循环使用视频帧

        bs_array = np.zeros([12], dtype=np.float32)
        bs_array[:6] = bs[0, :6]

        verts_frame_buffer = np.array(list_standard_vt)[frame_index, index_wrap, :2].copy() * 2 - 1

        rgba = renderModel_gl.render2cv(verts_frame_buffer, out_size=out_size, mat_world=mat_list[frame_index].T,
                                        bs_array=bs_array)
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

        rgb = rgb[::2, ::2, :]

        gl_tensor = torch.from_numpy(rgb / 255.).float().permute(2, 0, 1).unsqueeze(0)
        source_tensor = cv2.resize(list_standard_img[frame_index], (128, 128))
        source_tensor = torch.from_numpy(source_tensor / 255.).float().permute(2, 0, 1).unsqueeze(0)

        warped_img = renderModel_mini.interface(source_tensor, gl_tensor)

        image_numpy = warped_img.detach().squeeze(0).cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)

        x_min, y_min, x_max, y_max = list_source_crop_rect[frame_index]

        img_face = cv2.resize(image_numpy, (x_max - x_min, y_max - y_min))
        img_bg = list_video_img[frame_index]
        img_bg[y_min:y_max, x_min:x_max] = img_face

        videoWriter.write(img_bg[:, :, ::-1])
        frame_index += 1
    videoWriter.release()

    # 使用ffmpeg将处理后的视频和音频合并
    # 这里需要修改，因为音频是实时处理的，没有音频文件
    # os.system(
    #     "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p {}".format(save_path, wav_path, output_video_path))
    # os.remove(save_path)
    print("视频处理完成，请手动合并音频")
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 4:
        print("Usage: python demo_mini.py <video_path> <output_video_name>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Video path is set to: {video_path}")
    output_video_name = sys.argv[2]
    print(f"output video name is set to: {output_video_name}")
    try:
        model_name = sys.argv[3]
        print(f"model_name: {model_name}")
    except Exception:
        model_name = "render.pth"

    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)

    # 创建音频和特征向量队列
    audio_queue = queue.Queue()
    bs_queue = queue.Queue()

    # 加载音频模型
    Audio2FeatureModel = LoadAudioModel(r'checkpoint/lstm/lstm_model_epoch_325.pkl')

    # 创建音频处理线程
    audio_thread = threading.Thread(target=process_audio, args=(audio_queue, bs_queue, Audio2FeatureModel))
    audio_thread.start()

    # 创建视频处理线程
    video_thread = threading.Thread(target=process_video,
                                    args=(video_path, pkl_path, output_video_name, audio_queue, bs_queue))
    video_thread.start()

    # 主线程持续获取音频数据并放入队列
    try:
        while True:
            audio_data = get_audio_stream()
            audio_queue.put(audio_data)
    except KeyboardInterrupt:
        print("接收到键盘中断，停止音频处理")
        audio_queue.put(None)  # 发送结束信号给音频处理线程

    audio_thread.join()
    video_thread.join()
    print("所有线程已结束")


if __name__ == "__main__":
    main()