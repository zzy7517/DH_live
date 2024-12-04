import os
os.environ["kmp_duplicate_lib_ok"] = "true"
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os
from mini_live.obj.wrap_utils import index_wrap, index_edge_wrap
from mini_live.obj.obj_utils import generateRenderInfo, generateWrapModel
from talkingface.utils import crop_mouth, main_keypoints_index
current_dir = os.path.dirname(os.path.abspath(__file__))
import cv2
class RenderModel_gl:
    def __init__(self, window_size):
        self.window_size = window_size
        if not glfw.init():
            raise Exception("glfw can not be initialized!")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        print(window_size[0], window_size[1])
        self.window = glfw.create_window(window_size[0], window_size[1], "Face Render window", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")
        glfw.set_window_pos(self.window, 100, 100)
        glfw.make_context_current(self.window)
        # shader 设置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.program = compileProgram(compileShader(open(os.path.join(current_dir, "shader/prompt3.vsh")).readlines(), GL_VERTEX_SHADER),
                                       compileShader(open(os.path.join(current_dir, "shader/prompt3.fsh")).readlines(), GL_FRAGMENT_SHADER))
        self.VBO = glGenBuffers(1)
        self.render_verts = None
        self.render_face = None
        self.face_pts_mean = None

    def setContent(self, vertices_, face):
        glfw.make_context_current(self.window)
        self.render_verts = vertices_
        self.render_face = face
        glUseProgram(self.program)
        # set up vertex array object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.GenVBO(vertices_)
        self.GenEBO(face)

        # unbind VAO
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def GenEBO(self, face):
        self.indices = np.array(face, dtype=np.uint32)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def GenTexture(self, img, texture_index = GL_TEXTURE0):
        glfw.make_context_current(self.window)
        glActiveTexture(texture_index)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image_height, image_width = img.shape[:2]
        if len(img.shape) == 2:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, image_width, image_height, 0, GL_RED, GL_UNSIGNED_BYTE,
                         img.tobytes())
        elif img.shape[2] == 3:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
        elif img.shape[2] == 4:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
        else:
            print("Image Format not supported")
            exit(-1)

    def GenVBO(self, vertices_):
        glfw.make_context_current(self.window)
        vertices = np.array(vertices_, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(0))
        # 顶点纹理属性
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 5, ctypes.c_void_p(12))

    def render2cv(self, vertBuffer, out_size = (1000, 1000), mat_world=None, bs_array=None):
        glfw.make_context_current(self.window)
        # 设置正交投影矩阵
        # left = 0
        # right = standard_size
        # bottom = 0
        # top = standard_size
        # near = standard_size  # 近裁剪面距离
        # far = -standard_size  # 远裁剪面距离
        left = 0
        right = out_size[0]
        bottom = 0
        top = out_size[1]
        near = 1000  # 近裁剪面距离
        far = -1000  # 远裁剪面距离

        ortho_matrix = glm.ortho(left, right, bottom, top, near, far)
        # print("ortho_matrix: ", ortho_matrix)

        glUseProgram(self.program)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        # glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)  # 剔除背面
        glFrontFace(GL_CW)  # 通常顶点顺序是顺时针
        glClearColor(0.5, 0.5, 0.5, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # # 设置视口
        # glViewport(100, 0, self.window_size[0], self.window_size[1])


        glUniform1i(glGetUniformLocation(self.program, "texture_bs"), 0)
        glUniformMatrix4fv(glGetUniformLocation(self.program, "gWorld0"), 1, GL_FALSE, mat_world)
        glUniform1fv(glGetUniformLocation(self.program, "bsVec"), 12, bs_array.astype(np.float32))

        glUniform2fv(glGetUniformLocation(self.program, "vertBuffer"), 209, vertBuffer.astype(np.float32))

        glUniformMatrix4fv(glGetUniformLocation(self.program, "gProjection"), 1, GL_FALSE, glm.value_ptr(ortho_matrix))
        # bind VAO
        glBindVertexArray(self.vao)
        # draw
        glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_INT, None)
        # unbind VAO
        glBindVertexArray(0)

        glfw.swap_buffers(self.window)
        glReadBuffer(GL_FRONT)
        # 从缓冲区中的读出的数据是字节数组
        data = glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGBA, GL_UNSIGNED_BYTE, outputType=None)
        rgb = data.reshape(self.window_size[1], self.window_size[0], -1).astype(np.uint8)
        return rgb

def create_render_model(out_size = (384, 384), floor = 5):
    renderModel_gl = RenderModel_gl(out_size)

    image2 = cv2.imread(os.path.join(current_dir, "bs_texture.png"))

    image3 = np.zeros([12,256,3], dtype = np.uint8)
    image3[:, :len(index_wrap)] = image2[:,index_wrap]
    # print(image2.shape)
    # exit(-1)

    renderModel_gl.GenTexture(image3, GL_TEXTURE0)

    render_verts, render_face = generateRenderInfo()
    wrapModel_verts,wrapModel_face = generateWrapModel()

    renderModel_gl.setContent(wrapModel_verts, wrapModel_face)
    renderModel_gl.render_verts = render_verts
    renderModel_gl.render_face = render_face
    renderModel_gl.face_pts_mean = render_verts[:478, :3].copy()
    return renderModel_gl

# 示例使用
if __name__ == "__main__":
    import pickle
    import cv2
    import time
    import numpy as np
    import glob
    import random
    from OpenGL.GL import *
    import os

    import torch
    from talkingface.model_utils import LoadAudioModel,Audio2bs
    from talkingface.data.few_shot_dataset import get_image

    Audio2FeatureModel = LoadAudioModel(r'../checkpoint\lstm/lstm_model_epoch_325.pkl')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mouth_fusion_mask = cv2.imread("mouth_fusion_mask.png").astype(float)/255.


    from talkingface.render_model_mini import RenderModel_Mini
    renderModel_mini = RenderModel_Mini()
    renderModel_mini.loadModel("../checkpoint/FreeFace_3ref/epoch_60.pth")

    start_time = time.time()
    standard_size = 320
    crop_rotio = [0.5, 0.5, 0.5, 0.5]
    out_w = int(standard_size*(crop_rotio[0] + crop_rotio[1]))
    out_h = int(standard_size*(crop_rotio[2] + crop_rotio[3]))
    out_size = (out_w, out_h)
    renderModel_gl = create_render_model((out_w, out_h), floor = 20)

    wrapModel,wrapModel_face = generateWrapModel()

    path = r"F:\C\AI\CV\DH008_few_shot/preparation_new"
    video_list = os.listdir(r"{}".format(path))
    print(video_list)
    for test_video in video_list[:15]:
        Path_output_pkl = "{}/{}/keypoint_rotate.pkl".format(path, test_video)
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)

        video_path = "{}/{}/circle.mp4".format(path, test_video)
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

            standard_img = get_image(frame, source_crop_rect, input_type="image", resize = standard_size)
            standard_v = get_image(source_pts, source_crop_rect, input_type="mediapipe", resize = standard_size)
            standard_vt = standard_v[:, :2] / standard_size

            list_video_img.append(frame)
            list_source_crop_rect.append(source_crop_rect)
            list_standard_img.append(standard_img)
            list_standard_v.append(standard_v)
            list_standard_vt.append(standard_vt)
        cap.release()

        renderModel_mini.reset_charactor(list_standard_img, np.array(list_standard_v)[:,main_keypoints_index])

        # 求平均人脸
        from talkingface.run_utils import calc_face_mat
        mat_list, _, face_pts_mean_personal_primer = calc_face_mat(np.array(list_standard_v), renderModel_gl.face_pts_mean)
        face_wrap_entity = wrapModel.copy()
        # 正规点
        face_wrap_entity[:len(index_wrap),:3] = face_pts_mean_personal_primer[index_wrap, :3]
        # 边缘点
        vert_mid = face_wrap_entity[:,:3][index_edge_wrap[:4] + index_edge_wrap[-4:]].mean(axis=0)
        for index_, jj in enumerate(index_edge_wrap):
            face_wrap_entity[len(index_wrap) + index_,:3] = face_wrap_entity[jj, :3] + (face_wrap_entity[jj, :3] - vert_mid) * (0.3 - abs(10-index_)/40)
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

        wav2_dir = glob.glob(r"F:\C\AI\CV\DH008_few_shot\wav/*.wav")
        # wav2_dir = ["F:/aaa.wav"]
        sss = random.randint(0, len(wav2_dir)-1)
        wavpath = wav2_dir[sss]
        bs_array = Audio2bs(wavpath, Audio2FeatureModel)[5:] * 0.5

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

            rgba = renderModel_gl.render2cv(verts_frame_buffer, out_size=out_size, mat_world=mat_list[frame_index].T, bs_array=bs)
            rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)


            face_mask = (rgba[:,:,2] > 200)|((rgba[:,:,2] < 100))
            face_mask = face_mask.astype(np.uint8)
            face_mask = np.concatenate([face_mask[:,:,np.newaxis],face_mask[:,:,np.newaxis],face_mask[:,:,np.newaxis]], axis = 2)


            wrap_rgb = rgba[:,:,:2]
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
            warped_img = warped_img * (1-face_mask) + rgba * face_mask

            warped_img2 = img0.copy()
            pad = 5
            warped_img2[pad:-pad,pad:-pad] = warped_img[pad:-pad,pad:-pad]
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

            videoWriter.write(img_bg[:,:,::-1])
        videoWriter.release()
        os.makedirs("output", exist_ok=True)
        val_video = "output/{}.mp4".format(task_id + "_2")

        wav_path = wavpath
        os.system(
            "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p {}".format(save_path, wav_path, val_video))
        os.remove(save_path)

    cv2.destroyAllWindows()