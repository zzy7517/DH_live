import torch
from torch import nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F
class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=2)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True,
                 sample_mode='nearest'):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu
        self.sample_mode = sample_mode

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode=self.sample_mode)
        out = self.conv(out)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

class ResBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class SameBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

def make_coordinate_grid_3d(spatial_size, type):
    '''
        generate 3D coordinate grid
    '''
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1,-1, 1).repeat(d,1, w)
    xx = x.view(1,1, -1).repeat(d,h, 1)
    zz = z.view(-1,1,1).repeat(1,h,w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed,zz

class AdaAT(nn.Module):
    '''
       AdaAT operator
    '''
    def __init__(self,  para_ch,feature_ch, cuda = True):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
                    nn.Linear(para_ch, feature_ch),
                    nn.Sigmoid()
                )
        self.rotation = nn.Sequential(
                nn.Linear(para_ch, feature_ch),
                nn.Tanh()
            )
        self.translation = nn.Sequential(
                nn.Linear(para_ch, 2 * feature_ch),
                nn.Tanh()
            )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.cuda = cuda
        self.f_dim = (20, 14, 18)
        if cuda:
            self.grid_xy, self.grid_z = make_coordinate_grid_3d(self.f_dim, torch.cuda.FloatTensor)
        else:
            self.grid_xy, self.grid_z = make_coordinate_grid_3d(self.f_dim, torch.FloatTensor)
            batch = 1
            self.grid_xy = self.grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            self.grid_z = self.grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)


    def forward(self, feature_map,para_code):
        # batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        # batch= feature_map.size(0)
        if self.cuda:
            batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
            # print(batch, d, h, w)
            grid_xy = self.grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1).view(batch, d, h*w, 2)
            grid_z = self.grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        else:
            batch = 1
            d, h, w = self.f_dim
            grid_xy = self.grid_xy.view(batch, d, h*w, 2)
            grid_z = self.grid_z
        # print((d, h, w), feature_map.type())
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        # print(scale.size(), scale)
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159#
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        # rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        # grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        # grid_xy = self.grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        # grid_z = self.grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).repeat(1, 1, h*w, 1)
        # print(scale.size(), scale)
        # rotation_matrix = rotation_matrix.unsqueeze(2).repeat(1, 1, h*w, 1, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).repeat(1, 1, h * w, 1)  # torch.Size([bs, 256, 4096, 4])
        translation = translation.unsqueeze(2).repeat(1, 1, h*w, 1)

        # trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        trans_grid = torch.matmul(rotation_matrix.view(batch, d, h * w, 2, 2), grid_xy.unsqueeze(-1))

        trans_grid = trans_grid.squeeze(-1) * scale + translation
        # print(trans_grid.view(batch, d, h, w, 2).size(), grid_z.unsqueeze(-1).size())
        # trans_grid = torch.matmul(rotation_matrix.view(batch, d, h * w, 2, 2), grid_xy.unsqueeze(-1)).squeeze(
        #     -1) * scale + translation
        # print(trans_grid.view(batch, d, h, w, 2).size(), grid_z.unsqueeze(-1).size())
        full_grid = torch.cat([trans_grid.view(batch, d, h, w, 2), grid_z.unsqueeze(-1)], -1)
        # print(full_grid.size(), full_grid)

        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        # print("trans_feature", trans_feature.size())
        return trans_feature

class DINet_mini(nn.Module):
    def __init__(self, source_channel,ref_channel, cuda = True):
        super(DINet_mini, self).__init__()
        f_dim = 20
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 12, kernel_size=3, padding=1),
            DownBlock2d(12, 12, kernel_size=3, padding=1),
            DownBlock2d(12, f_dim, kernel_size=3, padding=1),
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 16, kernel_size=3, padding=1),
            DownBlock2d(16, 16, kernel_size=3, padding=1),
            SameBlock2d(16, 16, kernel_size=3, padding=1),
            DownBlock2d(16, f_dim, kernel_size=3, padding=1),
            # DownBlock2d(ref_channel, 1, kernel_size=3, padding=1),
            # DownBlock2d(1, f_dim, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 16 →8
            DownBlock2d(f_dim*2, f_dim, kernel_size=3, padding=1),
            # 8 →4
            DownBlock2d(f_dim, f_dim, kernel_size=3, padding=1),
        )

        appearance_conv_list = []
        appearance_conv_list.append(
            nn.Sequential(
                ResBlock2d(f_dim, f_dim, 3, 1),
            )
        )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(f_dim, f_dim, cuda)
        self.out_conv = nn.Sequential(
            SameBlock2d(f_dim*2, f_dim, kernel_size=3, padding=1),
            UpBlock2d(f_dim, f_dim, kernel_size=3, padding=1),
            SameBlock2d(f_dim, 16, 3, 1),
            UpBlock2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)

    def ref_input(self, ref_img):
        ## reference image encoder
        self.ref_img = ref_img
        self.ref_in_feature = self.ref_in_conv(self.ref_img)

    def interface(self, source_img):
        self.source_img = source_img
        ## source image encoder
        source_in_feature = self.source_in_conv(self.source_img)
        img_para = self.trans_conv(torch.cat([source_in_feature, self.ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        trans_para = img_para
        ref_trans_feature = self.adaAT(self.ref_in_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[0](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        out = self.out_conv(merge_feature)
        return out

    def forward(self, ref_img, driving_img):
        self.driving_img = driving_img
        self.ref_input(ref_img)
        out = self.interface(self.driving_img)
        return out

if __name__ == "__main__":
    device = "cpu"
    import torch.nn.functional as F
    size = (54, 72)  # h, w
    model = DINet_mini(3, 6*3, cuda=device is "cuda")
    model.eval()
    model = model.to(device)
    driving_img = torch.zeros([1, 3, size[0], size[1]]).to(device)
    ref_img = torch.zeros([1, 6*3, size[0], size[1]]).to(device)
    from thop import profile
    from thop import clever_format

    flops, params = profile(model.to(device), inputs=(ref_img, driving_img))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)