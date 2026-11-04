import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

CODE = "swc-17/SparseDrive fbadea6 projects/configs/sparsedrive_small_stage2.py"
CAMERAS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
MINI_TRAIN = ("scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094",
              "scene-1100")
SOURCE_SIZE = (1600, 900)
INPUT_SIZE = (704, 256)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
EMBED_DIMS = 256
NUM_GROUPS = 8
NUM_LEVELS = 4
NUM_CAMS = 6
EGO_FUT_TS = 6
X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = range(11)
DET_ORDER = (["gnn", "norm", "deformable", "ffn", "norm", "refine"]
             + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"] * 5)[2:]
MAP_ORDER = (["gnn", "norm", "deformable", "ffn", "norm", "refine"]
             + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"] * 5)
MOTION_ORDER = ["temp_gnn", "gnn", "norm", "cross_gnn", "norm", "ffn", "norm"] * 3 + ["refine"]
DET_FIX_SCALE = [[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]]


def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(np.asarray(q, dtype=np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(record):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(record["rotation"])
    matrix[:3, 3] = np.array(record["translation"])
    return matrix


def augmentation():
    width, height = SOURCE_SIZE
    final_width, final_height = INPUT_SIZE
    resize = max(final_height / height, final_width / width)
    resize_dims = (int(width * resize), int(height * resize))
    new_width, new_height = resize_dims
    crop_h = int((1 - 0.0) * new_height) - final_height
    crop_w = int(max(0, new_width - final_width) / 2)
    crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)
    transform = np.eye(3)
    transform[:2, :2] *= resize
    transform[:2, 2] -= np.array(crop[:2])
    rotation = np.array([[np.cos(0.0), np.sin(0.0), 0], [-np.sin(0.0), np.cos(0.0), 0], [0, 0, 1]])
    center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    rotation[:2, 2] = -rotation[:2, :2] @ center + center
    transform = rotation @ transform
    extended = np.eye(4)
    extended[:3, :3] = transform
    return resize_dims, crop, extended


def load_image(path, resize_dims, crop):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    image = Image.fromarray(image).resize(resize_dims, Image.Resampling.BICUBIC).crop(crop)
    image = np.array(image).astype(np.float32)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
    cv2.subtract(image, np.float64(MEAN.reshape(1, -1)), image)
    cv2.multiply(image, 1 / np.float64(STD.reshape(1, -1)), image)
    return image.transpose(2, 0, 1)


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i - 1] < utimes[i] - utime):
        i -= 1
    return i


class NuScenes:
    def __init__(self, root, version="v1.0-mini"):
        self.root = Path(root)

        def table(name):
            return {row["token"]: row for row in json.loads((self.root / version / f"{name}.json").read_text())}

        self.scenes = {row["name"]: row for row in table("scene").values()}
        self.samples = table("sample")
        self.ego_poses = table("ego_pose")
        self.calibrated_sensors = table("calibrated_sensor")
        sensors = table("sensor")
        self.sample_data = {}
        for row in table("sample_data").values():
            if row["is_key_frame"]:
                channel = sensors[self.calibrated_sensors[row["calibrated_sensor_token"]]["sensor_token"]]["channel"]
                self.sample_data[(row["sample_token"], channel)] = row
        self.resize_dims, self.crop, self.image_transform = augmentation()
        self.can_bus = {}

    def scene_samples(self, name):
        tokens = []
        token = self.scenes[name]["first_sample_token"]
        while token:
            tokens.append(token)
            token = self.samples[token]["next"]
        return tokens

    def lidar_records(self, token):
        lidar = self.sample_data[(token, "LIDAR_TOP")]
        return self.calibrated_sensors[lidar["calibrated_sensor_token"]], self.ego_poses[lidar["ego_pose_token"]]

    def ego_status(self, scene, timestamp):
        if scene not in self.can_bus:
            messages = {}
            for name in ("pose", "steeranglefeedback"):
                path = self.root / "can_bus" / f"{scene}_{name}.json"
                messages[name] = json.loads(path.read_text()) if path.exists() else None
            self.can_bus[scene] = messages
        messages = self.can_bus[scene]
        if messages["pose"] is None or messages["steeranglefeedback"] is None:
            return np.zeros(10, dtype=np.float32)
        pose = messages["pose"][locate_message([m["utime"] for m in messages["pose"]], timestamp)]
        steer = messages["steeranglefeedback"][
            locate_message([m["utime"] for m in messages["steeranglefeedback"]], timestamp)]
        return np.array(list(pose["accel"]) + list(pose["rotation_rate"]) + list(pose["vel"]) + [steer["value"]],
                        dtype=np.float32)

    def ego_future(self, token):
        lidar_cs, lidar_pose = self.lidar_records(token)
        trajectory = np.zeros((EGO_FUT_TS + 1, 3))
        masks = np.zeros(EGO_FUT_TS + 1)
        current = token
        for i in range(EGO_FUT_TS + 1):
            cs, pose = self.lidar_records(current)
            trajectory[i] = (pose_matrix(pose) @ pose_matrix(cs))[:3, 3]
            masks[i] = 1
            if self.samples[current]["next"] == "":
                trajectory[i + 1:] = trajectory[i]
                break
            current = self.samples[current]["next"]
        trajectory = trajectory - np.array(lidar_pose["translation"])
        trajectory = np.dot(quaternion_matrix(lidar_pose["rotation"]).T, trajectory.T).T
        trajectory = trajectory - np.array(lidar_cs["translation"])
        trajectory = np.dot(quaternion_matrix(lidar_cs["rotation"]).T, trajectory.T).T
        if trajectory[-1][0] >= 2:
            command = np.array([1, 0, 0])
        elif trajectory[-1][0] <= -2:
            command = np.array([0, 1, 0])
        else:
            command = np.array([0, 0, 1])
        offsets = trajectory[1:] - trajectory[:-1]
        return offsets[:, :2].astype(np.float32), masks[1:].astype(np.float32), command.astype(np.float32)

    def frame(self, token):
        sample = self.samples[token]
        lidar_cs, lidar_pose = self.lidar_records(token)
        l2e_t = np.array(lidar_cs["translation"])
        l2e_r_mat = quaternion_matrix(lidar_cs["rotation"])
        e2g_t = np.array(lidar_pose["translation"])
        e2g_r_mat = quaternion_matrix(lidar_pose["rotation"])
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = l2e_r_mat
        lidar2ego[:3, 3] = l2e_t
        ego2global = np.eye(4)
        ego2global[:3, :3] = e2g_r_mat
        ego2global[:3, 3] = e2g_t
        lidar2global = ego2global @ lidar2ego
        images = []
        lidar2img = []
        for camera in CAMERAS:
            record = self.sample_data[(token, camera)]
            cs = self.calibrated_sensors[record["calibrated_sensor_token"]]
            pose = self.ego_poses[record["ego_pose_token"]]
            l2e_r_s_mat = quaternion_matrix(cs["rotation"])
            e2g_r_s_mat = quaternion_matrix(pose["rotation"])
            rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            translation = (np.array(cs["translation"]) @ e2g_r_s_mat.T + np.array(pose["translation"])) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            translation -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(
                l2e_r_mat).T
            lidar2cam_r = np.linalg.inv(rotation.T)
            lidar2cam_t = translation @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4)
            viewpad[:3, :3] = np.array(cs["camera_intrinsic"])
            lidar2img.append(self.image_transform @ (viewpad @ lidar2cam_rt.T))
            images.append(load_image(self.root / record["filename"], self.resize_dims, self.crop))
        scene = next(name for name, row in self.scenes.items() if row["token"] == sample["scene_token"])
        fut_trajs, fut_masks, command = self.ego_future(token)
        return {
            "token": token,
            "scene": scene,
            "timestamp": sample["timestamp"] / 1e6,
            "img": np.ascontiguousarray(np.stack(images)),
            "projection_mat": np.float32(np.stack(lidar2img)),
            "image_wh": np.array([INPUT_SIZE] * NUM_CAMS, dtype=np.float32),
            "T_global": lidar2global,
            "T_global_inv": np.linalg.inv(lidar2global),
            "ego_status": self.ego_status(scene, sample["timestamp"]),
            "gt_ego_fut_trajs": fut_trajs,
            "gt_ego_fut_masks": fut_masks,
            "gt_ego_fut_cmd": command,
        }


def model_inputs(frame):
    img = torch.from_numpy(frame["img"])[None]
    metas = {
        "timestamp": torch.tensor([frame["timestamp"]], dtype=torch.float64),
        "projection_mat": torch.from_numpy(frame["projection_mat"])[None],
        "image_wh": torch.from_numpy(frame["image_wh"])[None],
        "ego_status": torch.from_numpy(frame["ego_status"])[None],
        "gt_ego_fut_cmd": torch.from_numpy(frame["gt_ego_fut_cmd"])[None],
        "img_metas": [{"T_global": frame["T_global"], "T_global_inv": frame["T_global_inv"],
                       "timestamp": frame["timestamp"]}],
    }
    return img, metas


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = None
        if stride != 1 or inplanes != planes * 4:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride, bias=False),
                                            nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        identity = x if self.downsample is None else self.downsample(x)
        return F.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        inplanes = 64
        for i, (planes, blocks, stride) in enumerate(((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2))):
            layer = [Bottleneck(inplanes, planes, stride)]
            layer += [Bottleneck(planes * 4, planes, 1) for _ in range(blocks - 1)]
            setattr(self, f"layer{i + 1}", nn.Sequential(*layer))
            inplanes = planes * 4

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        outs = []
        for i in range(4):
            x = getattr(self, f"layer{i + 1}")(x)
            outs.append(x)
        return outs


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=EMBED_DIMS):
        super().__init__()
        self.lateral_convs = nn.ModuleList(ConvModule(c, out_channels, 1) for c in in_channels)
        self.fpn_convs = nn.ModuleList(ConvModule(out_channels, out_channels, 3) for _ in in_channels)

    def forward(self, inputs):
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[2:],
                                                              mode="nearest")
        return [conv(x) for conv, x in zip(self.fpn_convs, laterals)]


class DenseDepthNet(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, num_depth_layers=3):
        super().__init__()
        self.depth_layers = nn.ModuleList(nn.Conv2d(embed_dims, 1, 1) for _ in range(num_depth_layers))


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class Scale(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * self.scale


class Attention(nn.Module):
    def __init__(self, embed_dims, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dims, embed_dims))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dims))
        self.out_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, key, value, key_padding_mask=None):
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)
        bs, num_query, dims = q.shape
        num_key = k.shape[1]
        head_dims = dims // self.num_heads
        q = q.reshape(bs, num_query, self.num_heads, head_dims).transpose(1, 2)
        k = k.reshape(bs, num_key, self.num_heads, head_dims).transpose(1, 2)
        v = v.reshape(bs, num_key, self.num_heads, head_dims).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) * head_dims ** -0.5
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        context = (logits.softmax(dim=-1) @ v).transpose(1, 2).reshape(bs, num_query, dims)
        return self.out_proj(context)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads=NUM_GROUPS):
        super().__init__()
        self.attn = Attention(embed_dims, num_heads)

    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        identity = query
        if key_pos is None and query_pos is not None and query_pos.shape == key.shape:
            key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        return identity + self.attn(query, key, value, key_padding_mask)


class AsymmetricFFN(nn.Module):
    def __init__(self, in_channels, embed_dims, feedforward_channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_channels)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(in_channels, feedforward_channels), nn.ReLU(inplace=True), nn.Dropout(0.0)),
            nn.Linear(feedforward_channels, embed_dims), nn.Dropout(0.0))
        self.identity_fc = nn.Identity() if feedforward_channels == embed_dims else nn.Linear(in_channels,
                                                                                             embed_dims)

    def forward(self, x):
        x = self.pre_norm(x)
        out = self.layers(x)
        return self.identity_fc(x) + out


def project_points(key_points, projection_mat, image_wh):
    pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
    points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
    points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
    return points_2d / image_wh[:, :, None, None]


def feature_maps_format(feature_maps):
    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []
    col_feats = []
    for feat in feature_maps:
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))
    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = torch.tensor([spatial_shape] * num_cams, dtype=torch.int64)
    scale_start_index = (spatial_shape[..., 0] * spatial_shape[..., 1]).flatten().cumsum(dim=0)
    scale_start_index = torch.cat([torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]])
    return [col_feats, spatial_shape, scale_start_index.reshape(num_cams, -1)]


def feature_map_level(feature_maps, cam, level):
    col_feats, spatial_shape, scale_start_index = feature_maps
    height, width = spatial_shape[cam, level].tolist()
    start = int(scale_start_index[cam, level])
    return col_feats[:, start:start + height * width].unflatten(1, (height, width)).permute(0, 3, 1, 2)


def deformable_aggregation(col_feats, spatial_shape, scale_start_index, sampling_location, weights):
    bs, num_feat, num_embeds = col_feats.shape
    num_anchors, num_pts, num_cams = sampling_location.shape[1:4]
    num_scale, num_groups = weights.shape[-2:]
    values = col_feats.reshape(-1, num_embeds)
    loc_w = sampling_location[..., 0]
    loc_h = sampling_location[..., 1]
    valid = (loc_w > 0) & (loc_w < 1) & (loc_h > 0) & (loc_h < 1)
    batch_index, anchor_index, pts_index, cam_index = torch.nonzero(valid, as_tuple=True)
    loc_w = loc_w[batch_index, anchor_index, pts_index, cam_index]
    loc_h = loc_h[batch_index, anchor_index, pts_index, cam_index]
    output = col_feats.new_zeros(bs * num_anchors, num_groups, num_embeds // num_groups)
    for scale_index in range(num_scale):
        h = spatial_shape[cam_index, scale_index, 0]
        w = spatial_shape[cam_index, scale_index, 1]
        value_offset = batch_index * num_feat + scale_start_index[cam_index, scale_index]
        h_im = loc_h * h - 0.5
        w_im = loc_w * w - 0.5
        h_low = torch.floor(h_im)
        w_low = torch.floor(w_im)
        lh = h_im - h_low
        lw = w_im - w_low
        hh = 1 - lh
        hw = 1 - lw
        h_low = h_low.long()
        w_low = w_low.long()
        sampled = None
        for dh, dw, coefficient in ((0, 0, hh * hw), (0, 1, hh * lw), (1, 0, lh * hw), (1, 1, lh * lw)):
            h_corner = h_low + dh
            w_corner = w_low + dw
            inside = (h_corner >= 0) & (h_corner <= h - 1) & (w_corner >= 0) & (w_corner <= w - 1)
            index = value_offset + h_corner.clamp(min=0) * w + w_corner.clamp(min=0)
            index = torch.where(inside, index, 0)
            term = values.index_select(0, index) * torch.where(inside, coefficient, 0)[:, None]
            sampled = term if sampled is None else sampled + term
        weight = weights[batch_index, anchor_index, pts_index, cam_index, scale_index]
        output.index_add_(0, batch_index * num_anchors + anchor_index,
                          sampled.view(-1, num_groups, num_embeds // num_groups) * weight[..., None])
    return output.reshape(bs, num_anchors, num_embeds)


class DeformableFeatureAggregation(nn.Module):
    def __init__(self, kps_generator, embed_dims=EMBED_DIMS, num_groups=NUM_GROUPS, num_levels=NUM_LEVELS,
                 num_cams=NUM_CAMS):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.kps_generator = kps_generator
        self.num_pts = kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.camera_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)

    def get_weights(self, instance_feature, anchor_embed, metas):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        camera_embed = self.camera_encoder(metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1))
        feature = feature[:, :, None] + camera_embed[:, None]
        return (self.weights_fc(feature).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
                .reshape(bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups))

    def forward(self, instance_feature, anchor, anchor_embed, feature_maps, metas):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self.get_weights(instance_feature, anchor_embed, metas)
        points_2d = (project_points(key_points, metas["projection_mat"], metas["image_wh"])
                     .permute(0, 2, 3, 1, 4).reshape(bs, num_anchor, self.num_pts, self.num_cams, 2))
        weights = (weights.permute(0, 1, 4, 2, 3, 5).contiguous()
                   .reshape(bs, num_anchor, self.num_pts, self.num_cams, self.num_levels, self.num_groups))
        features = deformable_aggregation(*feature_maps, points_2d, weights).reshape(bs, num_anchor, self.embed_dims)
        return torch.cat([self.output_proj(features), instance_feature], dim=-1)


class SparseBox3DKeyPointsGenerator(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, num_learnable_pts=0, num_fix_pts=1):
        super().__init__()
        self.num_learnable_pts = num_learnable_pts
        self.fix_scale = nn.Parameter(torch.zeros(num_fix_pts, 3), requires_grad=False)
        self.num_pts = num_fix_pts + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)

    def forward(self, anchor, instance_feature):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale * size
        learnable_scale = (self.learnable_fc(instance_feature).reshape(bs, num_anchor, self.num_learnable_pts, 3)
                           .sigmoid() - 0.5)
        key_points = torch.cat([key_points, learnable_scale * size], dim=-2)
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None]).squeeze(-1)
        return key_points + anchor[..., None, [X, Y, Z]]

    @staticmethod
    def anchor_projection(anchor, T_src2dst_list, time_intervals=None):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(T_src2dst_list[i].to(dtype=anchor.dtype), dim=1)
            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                translation = vel.transpose(0, -1) * time_intervals[i]
                translation = translation.transpose(0, -1)
                center = center - translation
            center = torch.matmul(T_src2dst[..., :3, :3], center[..., None]).squeeze(dim=-1) + T_src2dst[..., :3, 3]
            size = anchor[..., [W, L, H]]
            yaw = torch.matmul(T_src2dst[..., :2, :2], anchor[..., [COS_YAW, SIN_YAW], None]).squeeze(-1)
            yaw = yaw[..., [1, 0]]
            vel = torch.matmul(T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]).squeeze(-1)
            dst_anchors.append(torch.cat([center, size, yaw, vel], dim=-1))
        return dst_anchors


class SparsePoint3DKeyPointsGenerator(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, num_sample=20, num_learnable_pts=0, fix_height=(0,),
                 ground_height=0):
        super().__init__()
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, self.num_pts * 2)
        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

    def forward(self, anchor, instance_feature):
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = self.learnable_fc(instance_feature).reshape(bs, num_anchor, self.num_sample, len(self.fix_height),
                                                             self.num_learnable_pts, 2)
        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [key_points, key_points.new_full(key_points.shape[:-1] + (1,), fill_value=self.ground_height)], dim=-1)
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        return key_points.flatten(2, 4)

    def anchor_projection(self, anchor, T_src2dst_list, time_intervals=None):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            bs, num_anchor, _ = anchor.shape
            dst_anchor = anchor.clone().reshape(bs, num_anchor, self.num_sample, -1).flatten(1, 2)
            T_src2dst = torch.unsqueeze(T_src2dst_list[i].to(dtype=anchor.dtype), dim=1)
            dst_anchor = (torch.matmul(T_src2dst[..., :2, :2], dst_anchor[..., None]).squeeze(dim=-1)
                          + T_src2dst[..., :2, 3])
            dst_anchors.append(dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(2, 3))
        return dst_anchors


class SparseBox3DEncoder(nn.Module):
    def __init__(self, embed_dims=(128, 32, 32, 64), vel_dims=3, out_loops=4):
        super().__init__()
        self.vel_dims = vel_dims
        self.pos_fc = nn.Sequential(*linear_relu_ln(embed_dims[0], 1, out_loops, 3))
        self.size_fc = nn.Sequential(*linear_relu_ln(embed_dims[1], 1, out_loops, 3))
        self.yaw_fc = nn.Sequential(*linear_relu_ln(embed_dims[2], 1, out_loops, 2))
        self.vel_fc = nn.Sequential(*linear_relu_ln(embed_dims[3], 1, out_loops, vel_dims))

    def forward(self, box_3d):
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])
        size_feat = self.size_fc(box_3d[..., [W, L, H]])
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]])
        vel_feat = self.vel_fc(box_3d[..., VX:VX + self.vel_dims])
        return torch.cat([pos_feat, size_feat, yaw_feat, vel_feat], dim=-1)


class SparsePoint3DEncoder(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, num_sample=20, coords_dim=2):
        super().__init__()
        self.pos_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, num_sample * coords_dim))

    def forward(self, anchor):
        return self.pos_fc(anchor)


class SparseBox3DRefinementModule(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, output_dim=11, num_cls=10):
        super().__init__()
        self.output_dim = output_dim
        self.refine_state = [X, Y, Z, W, L, H, SIN_YAW, COS_YAW]
        self.layers = nn.Sequential(*linear_relu_ln(embed_dims, 2, 2), nn.Linear(embed_dims, output_dim),
                                    Scale(output_dim))
        self.cls_layers = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, num_cls))
        self.quality_layers = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 2))

    def forward(self, instance_feature, anchor, anchor_embed, time_interval):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        output[..., self.refine_state] = output[..., self.refine_state] + anchor[..., self.refine_state]
        translation = torch.transpose(output[..., VX:], 0, -1)
        velocity = torch.transpose(translation / time_interval, 0, -1)
        output[..., VX:] = velocity + anchor[..., VX:]
        cls = self.cls_layers(instance_feature)
        quality = self.quality_layers(feature)
        return output, cls, quality


class SparsePoint3DRefinementModule(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, num_sample=20, coords_dim=2, num_cls=3):
        super().__init__()
        output_dim = num_sample * coords_dim
        self.layers = nn.Sequential(*linear_relu_ln(embed_dims, 2, 2), nn.Linear(embed_dims, output_dim),
                                    Scale(output_dim))
        self.cls_layers = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, num_cls))

    def forward(self, instance_feature, anchor, anchor_embed, time_interval):
        output = self.layers(instance_feature + anchor_embed) + anchor
        return output, self.cls_layers(instance_feature), None


def topk(confidence, k, *inputs):
    bs, n = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * n).reshape(-1)
    return confidence, [x.flatten(end_dim=1)[indices].reshape(bs, k, -1) for x in inputs]


class InstanceBank(nn.Module):
    def __init__(self, num_anchor, anchor_dims, anchor_handler, num_temp_instances, embed_dims=EMBED_DIMS,
                 default_time_interval=0.5, confidence_decay=0.6, max_time_interval=2):
        super().__init__()
        self.num_anchor = num_anchor
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.anchor_handler = anchor_handler
        self.anchor = nn.Parameter(torch.zeros(num_anchor, anchor_dims))
        self.instance_feature = nn.Parameter(torch.zeros(num_anchor, embed_dims))
        self.reset()

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas):
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            time_interval = (metas["timestamp"] - self.metas["timestamp"]).to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = self.cached_anchor.new_tensor(np.stack(
                [x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"] for i, x in enumerate(metas["img_metas"])]))
            self.cached_anchor = self.anchor_handler.anchor_projection(self.cached_anchor, [T_temp2cur],
                                                                       time_intervals=[-time_interval])[0]
            time_interval = torch.where(torch.logical_and(time_interval != 0, self.mask), time_interval,
                                        time_interval.new_tensor(self.default_time_interval))
        else:
            self.reset()
            time_interval = instance_feature.new_tensor([self.default_time_interval] * batch_size)
        return instance_feature, anchor, self.cached_feature, self.cached_anchor, time_interval

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor
        n = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(confidence, n, instance_feature, anchor)
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        instance_feature = torch.where(self.mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(self.mask[:, None], self.confidence, self.confidence.new_tensor(0))
        if self.instance_id is not None:
            self.instance_id = torch.where(self.mask[:, None], self.instance_id, self.instance_id.new_tensor(-1))
        return instance_feature, anchor

    def cache(self, instance_feature, anchor, confidence, metas):
        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, :self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay, confidence[:, :self.num_temp_instances])
        self.temp_confidence = confidence
        self.confidence, (self.cached_feature, self.cached_anchor) = topk(
            confidence, self.num_temp_instances, instance_feature, anchor)

    def get_instance_id(self, confidence):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()
        if self.instance_id is not None and self.instance_id.shape[0] == instance_id.shape[0]:
            instance_id[:, :self.instance_id.shape[1]] = self.instance_id
        mask = instance_id < 0
        num_new_instance = mask.sum()
        instance_id[torch.where(mask)] = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        self.prev_id += num_new_instance
        temp_conf = self.temp_confidence
        ids = topk(temp_conf, self.num_temp_instances, instance_id)[1][0].squeeze(dim=-1)
        self.instance_id = F.pad(ids, (0, self.num_anchor - self.num_temp_instances), value=-1)
        return instance_id


class Sparse4DHead(nn.Module):
    def __init__(self, instance_bank, anchor_encoder, operation_order, build, decouple_attn,
                 num_single_frame_decoder=1, with_instance_id=True):
        super().__init__()
        self.instance_bank = instance_bank
        self.anchor_encoder = anchor_encoder
        self.operation_order = operation_order
        self.num_single_frame_decoder = num_single_frame_decoder
        self.with_instance_id = with_instance_id
        self.decouple_attn = decouple_attn
        self.layers = nn.ModuleList(build(op) for op in operation_order)
        if decouple_attn:
            self.fc_before = nn.Linear(EMBED_DIMS, EMBED_DIMS * 2, bias=False)
            self.fc_after = nn.Linear(EMBED_DIMS * 2, EMBED_DIMS, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def graph_model(self, index, query, key=None, value=None, query_pos=None, key_pos=None):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(self.layers[index](query, key, value, query_pos=query_pos, key_pos=key_pos))

    def forward(self, feature_maps, metas):
        batch_size = feature_maps[0].shape[0]
        instance_feature, anchor, temp_instance_feature, temp_anchor, time_interval = self.instance_bank.get(
            batch_size, metas)
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(temp_anchor) if temp_anchor is not None else None
        prediction = []
        classification = []
        quality = []
        cls = None
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                instance_feature = self.graph_model(i, instance_feature, temp_instance_feature,
                                                    temp_instance_feature, query_pos=anchor_embed,
                                                    key_pos=temp_anchor_embed)
            elif op == "gnn":
                instance_feature = self.graph_model(i, instance_feature, value=instance_feature,
                                                    query_pos=anchor_embed)
            elif op in ("norm", "ffn"):
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](instance_feature, anchor, anchor_embed, feature_maps, metas)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](instance_feature, anchor, anchor_embed, time_interval)
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(instance_feature, anchor, cls)
                anchor_embed = self.anchor_encoder(anchor)
                if len(prediction) > self.num_single_frame_decoder and temp_anchor_embed is not None:
                    temp_anchor_embed = anchor_embed[:, :self.instance_bank.num_temp_instances]
            else:
                raise ValueError(op)
        output = {"classification": classification, "prediction": prediction, "quality": quality,
                  "instance_feature": instance_feature, "anchor_embed": anchor_embed}
        self.instance_bank.cache(instance_feature, anchor, cls, metas)
        if self.with_instance_id:
            output["instance_id"] = self.instance_bank.get_instance_id(cls)
        return output


def gen_sineembed_for_position(pos_tensor, hidden_dim=EMBED_DIMS):
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class InstanceQueue(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, queue_length=4, tracking_threshold=0.2,
                 feature_map_scale=(INPUT_SIZE[1] / 32, INPUT_SIZE[0] / 32)):
        super().__init__()
        self.queue_length = queue_length
        self.tracking_threshold = tracking_threshold
        kernel_size = tuple(int(x / 2) for x in feature_map_scale)
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(embed_dims),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(embed_dims),
            nn.ReLU(), nn.AvgPool2d(kernel_size))
        self.ego_anchor = nn.Parameter(torch.zeros(1, 11), requires_grad=False)
        self.reset()

    def reset(self):
        self.metas = None
        self.prev_instance_id = None
        self.prev_confidence = None
        self.period = None
        self.instance_feature_queue = []
        self.anchor_queue = []
        self.prev_ego_status = None
        self.ego_period = None
        self.ego_feature_queue = []
        self.ego_anchor_queue = []

    def get(self, det_output, feature_maps, metas, batch_size, mask, anchor_handler):
        if self.period is not None and batch_size == self.period.shape[0]:
            T_temp2cur = feature_maps[0].new_tensor(np.stack(
                [x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"] for i, x in enumerate(metas["img_metas"])]))
            for i in range(len(self.anchor_queue)):
                self.anchor_queue[i] = anchor_handler.anchor_projection(self.anchor_queue[i], [T_temp2cur])[0]
            for i in range(len(self.ego_anchor_queue)):
                self.ego_anchor_queue[i] = anchor_handler.anchor_projection(self.ego_anchor_queue[i], [T_temp2cur])[0]
        else:
            self.reset()
        self.prepare_motion(det_output)
        ego_feature, ego_anchor = self.prepare_planning(feature_maps, mask, batch_size)
        temp_instance_feature = torch.stack(self.instance_feature_queue, dim=2)
        temp_anchor = torch.stack(self.anchor_queue, dim=2)
        temp_ego_feature = torch.stack(self.ego_feature_queue, dim=2)
        temp_ego_anchor = torch.stack(self.ego_anchor_queue, dim=2)
        period = torch.cat([self.period, self.ego_period], dim=1)
        temp_instance_feature = torch.cat([temp_instance_feature, temp_ego_feature], dim=1)
        temp_anchor = torch.cat([temp_anchor, temp_ego_anchor], dim=1)
        num_agent = temp_anchor.shape[1]
        temp_mask = torch.arange(len(self.anchor_queue), 0, -1, device=temp_anchor.device)
        temp_mask = temp_mask[None, None].repeat((batch_size, num_agent, 1))
        temp_mask = torch.gt(temp_mask, period[..., None])
        return ego_feature, ego_anchor, temp_instance_feature, temp_anchor, temp_mask

    def prepare_motion(self, det_output):
        instance_feature = det_output["instance_feature"]
        det_anchors = det_output["prediction"][-1]
        if self.period is None:
            self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()
        else:
            instance_id = det_output["instance_id"]
            match = instance_id[..., None] == self.prev_instance_id[:, None]
            if self.tracking_threshold > 0:
                temp_mask = self.prev_confidence > self.tracking_threshold
                match = match * temp_mask.unsqueeze(1)
            for i in range(len(self.instance_feature_queue)):
                self.instance_feature_queue[i] = (match[..., None] * self.instance_feature_queue[i][:, None]).sum(
                    dim=2)
                self.anchor_queue[i] = (match[..., None] * self.anchor_queue[i][:, None]).sum(dim=2)
            self.period = (match * self.period[:, None]).sum(dim=2)
        self.instance_feature_queue.append(instance_feature)
        self.anchor_queue.append(det_anchors)
        self.period += 1
        if len(self.instance_feature_queue) > self.queue_length:
            self.instance_feature_queue.pop(0)
            self.anchor_queue.pop(0)
        self.period = torch.clip(self.period, 0, self.queue_length)

    def prepare_planning(self, feature_maps, mask, batch_size):
        ego_feature = self.ego_feature_encoder(feature_map_level(feature_maps, 0, -1))
        ego_feature = ego_feature.unsqueeze(1).squeeze(-1).squeeze(-1)
        ego_anchor = torch.tile(self.ego_anchor[None], (batch_size, 1, 1))
        if self.prev_ego_status is not None:
            prev_ego_status = torch.where(mask[:, None, None], self.prev_ego_status,
                                          self.prev_ego_status.new_tensor(0))
            ego_anchor[..., VY] = prev_ego_status[..., 6]
        if self.ego_period is None:
            self.ego_period = ego_feature.new_zeros((batch_size, 1)).long()
        else:
            self.ego_period = torch.where(mask[:, None], self.ego_period, self.ego_period.new_tensor(0))
        self.ego_feature_queue.append(ego_feature)
        self.ego_anchor_queue.append(ego_anchor)
        self.ego_period += 1
        if len(self.ego_feature_queue) > self.queue_length:
            self.ego_feature_queue.pop(0)
            self.ego_anchor_queue.pop(0)
        self.ego_period = torch.clip(self.ego_period, 0, self.queue_length)
        return ego_feature, ego_anchor

    def cache_motion(self, det_output, metas):
        self.metas = metas
        self.prev_confidence = det_output["classification"][-1].sigmoid().max(dim=-1).values
        self.prev_instance_id = det_output["instance_id"]

    def cache_planning(self, ego_feature, ego_status):
        self.prev_ego_status = ego_status
        self.ego_feature_queue[-1] = ego_feature


class MotionPlanningRefinementModule(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, fut_ts=12, fut_mode=6, ego_fut_ts=EGO_FUT_TS, ego_fut_mode=6):
        super().__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        def branch(output_dims):
            return nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims),
                                 nn.ReLU(), nn.Linear(embed_dims, output_dims))

        self.motion_cls_branch = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 1))
        self.motion_reg_branch = branch(fut_ts * 2)
        self.plan_cls_branch = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 1))
        self.plan_reg_branch = branch(ego_fut_ts * 2)
        self.plan_status_branch = branch(10)

    def forward(self, motion_query, plan_query, ego_feature, ego_anchor_embed):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status


class MotionPlanningHead(nn.Module):
    def __init__(self, embed_dims=EMBED_DIMS, num_det=50, num_map=10):
        super().__init__()
        self.operation_order = MOTION_ORDER
        self.num_det = num_det
        self.num_map = num_map
        self.instance_queue = InstanceQueue(embed_dims)

        def build(op):
            if op in ("temp_gnn", "gnn"):
                return MultiheadAttention(embed_dims * 2)
            if op == "cross_gnn":
                return MultiheadAttention(embed_dims)
            if op == "norm":
                return nn.LayerNorm(embed_dims)
            if op == "ffn":
                return AsymmetricFFN(embed_dims, embed_dims, embed_dims * 2)
            return MotionPlanningRefinementModule(embed_dims)

        self.layers = nn.ModuleList(build(op) for op in self.operation_order)
        self.fc_before = nn.Linear(embed_dims, embed_dims * 2, bias=False)
        self.fc_after = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.motion_anchor = nn.Parameter(torch.zeros(10, 6, 12, 2), requires_grad=False)
        self.motion_anchor_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 1),
                                                   nn.Linear(embed_dims, embed_dims))
        self.plan_anchor = nn.Parameter(torch.zeros(3, 6, EGO_FUT_TS, 2), requires_grad=False)
        self.plan_anchor_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 1), nn.Linear(embed_dims, embed_dims))

    def get_motion_anchor(self, classification, prediction):
        motion_anchor = self.motion_anchor[classification.argmax(dim=-1)]
        yaw = torch.atan2(prediction[..., SIN_YAW], prediction[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack([torch.stack([cos_yaw, sin_yaw]), torch.stack([-sin_yaw, cos_yaw])])
        return torch.einsum("abcij,jkab->abcik", motion_anchor, rot_mat_T)

    def graph_model(self, index, query, key=None, value=None, query_pos=None, key_pos=None, key_padding_mask=None):
        query = torch.cat([query, query_pos], dim=-1)
        if key is not None:
            key = torch.cat([key, key_pos], dim=-1)
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(self.layers[index](query, key, value, key_padding_mask=key_padding_mask))

    def forward(self, det_output, map_output, feature_maps, metas, anchor_encoder, mask, anchor_handler):
        instance_feature = det_output["instance_feature"]
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(det_confidence, self.num_det, instance_feature,
                                                                     anchor_embed)
        map_confidence = map_output["classification"][-1].sigmoid().max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_output["instance_feature"], map_output["anchor_embed"])
        bs, num_anchor, dim = instance_feature.shape
        ego_feature, ego_anchor, temp_instance_feature, temp_anchor, temp_mask = self.instance_queue.get(
            det_output, feature_maps, metas, bs, mask, anchor_handler)
        ego_anchor_embed = anchor_encoder(ego_anchor)
        temp_anchor_embed = anchor_encoder(temp_anchor)
        temp_instance_feature = temp_instance_feature.flatten(0, 1)
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)
        temp_mask = temp_mask.flatten(0, 1)
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        plan_anchor = torch.tile(self.plan_anchor[None], (bs, 1, 1, 1, 1))
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)
        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        plan_status = None
        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                instance_feature = self.graph_model(
                    i, instance_feature.flatten(0, 1).unsqueeze(1), temp_instance_feature, temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1), key_pos=temp_anchor_embed,
                    key_padding_mask=temp_mask)
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)
            elif op == "gnn":
                instance_feature = self.graph_model(i, instance_feature, instance_feature_selected,
                                                    instance_feature_selected, query_pos=anchor_embed,
                                                    key_pos=anchor_embed_selected)
            elif op in ("norm", "ffn"):
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                instance_feature = self.layers[i](instance_feature, key=map_instance_feature_selected,
                                                  query_pos=anchor_embed, key_pos=map_anchor_embed_selected)
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
                motion_cls, motion_reg, plan_cls, plan_reg, plan_status = self.layers[i](
                    motion_query, plan_query, instance_feature[:, num_anchor:], anchor_embed[:, num_anchor:])
                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)
        self.instance_queue.cache_motion(det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)
        motion_output = {"classification": motion_classification, "prediction": motion_prediction,
                         "period": self.instance_queue.period, "anchor_queue": self.instance_queue.anchor_queue}
        planning_output = {"classification": planning_classification, "prediction": planning_prediction,
                           "status": planning_status, "period": self.instance_queue.ego_period,
                           "anchor_queue": self.instance_queue.ego_anchor_queue}
        return motion_output, planning_output


def det_head():
    def build(op):
        if op in ("temp_gnn", "gnn"):
            return MultiheadAttention(EMBED_DIMS * 2)
        if op == "norm":
            return nn.LayerNorm(EMBED_DIMS)
        if op == "ffn":
            return AsymmetricFFN(EMBED_DIMS * 2, EMBED_DIMS, EMBED_DIMS * 4)
        if op == "deformable":
            return DeformableFeatureAggregation(SparseBox3DKeyPointsGenerator(
                num_learnable_pts=6, num_fix_pts=len(DET_FIX_SCALE)))
        return SparseBox3DRefinementModule()

    bank = InstanceBank(900, 11, SparseBox3DKeyPointsGenerator(), num_temp_instances=600)
    return Sparse4DHead(bank, SparseBox3DEncoder(), DET_ORDER, build, decouple_attn=True)


def map_head():
    def build(op):
        if op in ("temp_gnn", "gnn"):
            return MultiheadAttention(EMBED_DIMS)
        if op == "norm":
            return nn.LayerNorm(EMBED_DIMS)
        if op == "ffn":
            return AsymmetricFFN(EMBED_DIMS * 2, EMBED_DIMS, EMBED_DIMS * 4)
        if op == "deformable":
            return DeformableFeatureAggregation(SparsePoint3DKeyPointsGenerator(
                num_sample=20, num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1), ground_height=-1.84023))
        return SparsePoint3DRefinementModule()

    bank = InstanceBank(100, 40, SparsePoint3DKeyPointsGenerator(), num_temp_instances=33)
    return Sparse4DHead(bank, SparsePoint3DEncoder(), MAP_ORDER, build, decouple_attn=False, with_instance_id=False)


class SparseDriveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_head = det_head()
        self.map_head = map_head()
        self.motion_plan_head = MotionPlanningHead()

    def forward(self, feature_maps, metas):
        det_output = self.det_head(feature_maps, metas)
        map_output = self.map_head(feature_maps, metas)
        motion_output, planning_output = self.motion_plan_head(
            det_output, map_output, feature_maps, metas, self.det_head.anchor_encoder,
            self.det_head.instance_bank.mask, self.det_head.instance_bank.anchor_handler)
        return det_output, map_output, motion_output, planning_output


class SparseDrive(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ResNet()
        self.img_neck = FPN()
        self.depth_branch = DenseDepthNet()
        self.head = SparseDriveHead()

    def reset(self):
        self.head.det_head.instance_bank.reset()
        self.head.map_head.instance_bank.reset()
        self.head.motion_plan_head.instance_queue.reset()

    def extract_feat(self, img):
        bs, num_cams = img.shape[:2]
        feature_maps = self.img_neck(self.img_backbone(img.flatten(end_dim=1)))
        return [torch.reshape(feat, (bs, num_cams) + feat.shape[1:]) for feat in feature_maps]

    def forward(self, img, metas):
        return self.head(feature_maps_format(self.extract_feat(img)), metas)


def load_model(checkpoint):
    model = SparseDrive()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def fold_batchnorm(model):
    pairs = [(model.img_backbone, "conv1", "bn1")]
    for module in model.img_backbone.modules():
        if isinstance(module, Bottleneck):
            pairs += [(module, f"conv{i}", f"bn{i}") for i in (1, 2, 3)]
            if module.downsample is not None:
                pairs.append((module.downsample, "0", "1"))
    encoder = model.head.motion_plan_head.instance_queue.ego_feature_encoder
    pairs += [(encoder, "0", "1"), (encoder, "2", "3")]
    folded = []
    names = {module: name for name, module in model.named_modules()}
    for parent, conv_name, bn_name in pairs:
        conv = getattr(parent, conv_name)
        bn = getattr(parent, bn_name)
        bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight = nn.Parameter(conv.weight * factor.reshape(-1, 1, 1, 1))
        conv.bias = nn.Parameter((bias - bn.running_mean) * factor + bn.bias)
        setattr(parent, bn_name, nn.Identity())
        folded.append(names[conv])
    return folded
