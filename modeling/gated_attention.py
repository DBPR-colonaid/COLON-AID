import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
from einops import rearrange
from modeling.ct_pos_embedding import ScanIDEmbedding, PositionEmbeddingMLP
from modeling.constants import FEATURE_KEYS_CT, FEATURE_KEYS_nonCT
from modeling.constants import FEATURE_DIMS_V3 as FEATURE_DIMS


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Att_Head(nn.Module):
    def __init__(self,FEATURE_DIM,ATT_IM_DIM):
        super(Att_Head, self).__init__()

        self.fc1 = nn.Linear(FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class GatedAttentionSummarizer(nn.Module):
    def __init__(self, gate=True, dim_in=768, dim_out=768, dim_mid=384, dropout=True, n_classes=None):
        super(GatedAttentionSummarizer, self).__init__()
        # self.size_dict = {'xs': [384, 256, 256], "small": [768, 512, 256], "big": [1024, 512, 384], 'large': [2048, 1024, 512]}
        size = [dim_in, dim_out, dim_mid]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes) if n_classes is not None else nn.Identity()

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def forward(self, h, attention_only=False):
        device = h.device
        #h=h.squeeze(0)
        A, h = self.attention_net(h)  # NxK, Nxd
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)  # A: K (K=1) * N h: N * 512 => M: 1 * 512
        logits = self.classifiers(M)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        result = {
            'cls': logits,
            'attention_raw': A_raw,
            'M': M
        }
        return result


# main classes
class GatedCTEventFormer(nn.Module):
    """
    Input:
        CT Event
    Process:
    Output:
    """

    def __init__(
            self,
            *,
            gated_summarizer: Module,
            dim=768,
            move_attn_cpu=True,
            config=None,
            max_unique_scans=64,
    ):
        super().__init__()
        self.gated_summarizer = gated_summarizer
        self.dim = dim
        self.move_attn_cpu = move_attn_cpu
        self.config = config

        img_feature_keys = [k for k in config.img_feature_keys if k in FEATURE_KEYS_CT]  # only use CT features
        n_keys = len(img_feature_keys)

        # Linear layers for image features
        self.img_feature_linears = nn.ModuleDict({
            key: nn.Linear(FEATURE_DIMS[key], dim)
            for key in img_feature_keys
        })

        self.scanid_embd = ScanIDEmbedding(dim, max_unique_scans=max_unique_scans)
        inner_dim = min((3 + dim) // 2, dim)
        self.pos_emb = PositionEmbeddingMLP(3, inner_dim, dim, normalizer_in_mm=150, dropout=0.2)
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.hidden_size)
        )

    def return_zero_condition(self, event):
        if not event['img_features'] or not event['img_features'][list(event['img_features'].keys())[0]]:
            self.config.logger.debug(f'Event {event["pacs"]} {event["uid"]} on {event.get("date")} has no image features.')
            return True
        return False

    def forward(self, event):
        if self.return_zero_condition(event):
            cls = torch.zeros(1, self.config.hidden_size, device=self.pos_emb.layer1.weight.device, requires_grad=True)
            attn = []
            res = {'cls': cls, 'attn_list': attn}
            return res

        assert event['content'] == 'CT', 'Only CT events are supported in CTEventFormer.'
        cls_list = []
        attn_list = []
        # Process each image feature key
        for key in self.img_feature_linears:
            if key not in event['img_features'] or event['img_features'][key] is None:
                # print(f'Event {event["event_type"]} on {event.get("date")} has no image features of type {key}.')
                # cls_list.append(torch.zeros(1, self.dim, device=self.key_fusion[1].weight.device))
                raise ValueError(f'Event {event} on {event.get("date")} has no image features of type {key}.')

            data = event['img_features'][key]
            if not data:
                raise ValueError(f'Event {event} on {event.get("date")} empty image feature for {key}.')
            features, meta = data[0]['features'], data[0]['meta']  # only one feature tensor per CT event per key
            if features.dim() == 2:  # n, 4096 from FMCIB
                features = features[..., None, None, None]  # n_patches, feat_dim, 1, 1, 1
            features = rearrange(features, 'f c d h w -> () f d h w c')  # 1, n_patches, d, h, w, feat_dim
            features = self.img_feature_linears[key](features)  # 1, n_patches, d, h, w, dim
            device = features.device

            scan_ids = meta[:, 0]
            coords = meta[:, 1:4].astype(np.float32)  # (n_patches, 3)
            coords = torch.from_numpy(coords).to(device, dtype=features.dtype)
            organ_types = meta[:, 4]

            scanid_embd = self.scanid_embd(scan_ids)  # (n_patches, dim)
            pos_embd = self.pos_emb(coords)  # (n_patches, dim)
            features = features + scanid_embd[None, :, None, None, None, :] + pos_embd[None, :, None, None, None,
                                                                              :]  # (1, n_patches, 8, 8, 8, 768)
            _, n, d, h, w, _ = features.shape
            features = rearrange(features, '() n d h w c -> (n d h w) c')  # (1, n_patches * 8 * 8 * 8, 768)

            # import pdb; pdb.set_trace()

            res_dict = self.gated_summarizer(features)  # {'cls': (1, dim), 'attention_raw': (1, n_patches)}
            cls = res_dict['cls']
            attn = res_dict['attention_raw']
            if self.move_attn_cpu:
                attn = attn.cpu().unsqueeze(dim=1)  # 1, 1, (n d h w)
                attn = rearrange(attn, '1 1 (n d h w) -> 1 1 n d h w', n=n, d=d, h=h, w=w)
            cls_list.append(cls)
            attn_list.append(attn)
        cls = sum(cls_list) / len(cls_list)
        cls = self.linear(cls)
        res = {
            'cls': cls,
            'attn_list': attn_list
        }
        return res


def get_gated_attnetion_ct(config):
    dim_in = config.model.dim_ct_former
    dim_out = config.model.dim_ct_former
    dim_mid = 384
    gated_summarizer = GatedAttentionSummarizer(gate=True, dim_in=dim_in, dim_out=dim_out, dim_mid=dim_mid,
                                                dropout=True, n_classes=None)
    model = GatedCTEventFormer(gated_summarizer=gated_summarizer, dim=dim_out, move_attn_cpu=config.model.move_attn_cpu,
                               config=config)
    return model


def split_video(video, max_frames, frame_axis=2):
    n_slices = video.shape[frame_axis]
    num_splits = int(np.ceil(n_slices / max_frames))
    split_size = n_slices // num_splits

    frames = video.split(split_size, dim = frame_axis)
    return list(frames)


class GatedVideoEventFormer(nn.Module):
    """
    VideoEventFormer class
    Input:
        2D Event
    Process:
    Output:
    """
    def __init__(
        self,
        *,
        gated_summarizer: Module,
        dim=768,
        move_attn_cpu=True,
        max_frames=256,
        config=None,
    ):
        super().__init__()
        self.gated_summarizer = gated_summarizer
        self.max_frames = max_frames
        self.dim = dim
        self.move_attn_cpu = move_attn_cpu
        self.config = config

        img_feature_keys = [k for k in config.img_feature_keys if k in FEATURE_KEYS_nonCT]  # only use non-CT features

        # Linear layers for image features
        self.img_feature_linears = nn.ModuleDict({
            key: nn.Linear(FEATURE_DIMS[key], dim)
            for key in img_feature_keys
        })
        self.linear = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.hidden_size)
        )

    def return_zero_condition(self, event):
        if not event['img_features'] or not event['img_features'][list(event['img_features'].keys())[0]]:
            self.config.logger.info(f'Event {event["event_type"]} on {event.get("date")} has no image features.')
            return True
        return False

    def forward(self, event):
        if self.return_zero_condition(event):
            # if event['content'] == 'MR':
            #     import pdb; pdb.set_trace()
            cls = torch.zeros(1, self.config.hidden_size, device=self.gated_summarizer.attention_net[0].weight.device, requires_grad=True)
            attn = []
            res = {
                'cls': cls,
                'attn_list': attn
            }
            return res

        list_of_tensors = event['img_features']['SAMMed2D']  # I take only SAMMed2D
        list_of_tensors = [x['features'] for x in list_of_tensors]  # each of shape (1, c, f, h, w)

        max_series = 32 if list_of_tensors[0].shape[3] == 4 else 4  # MR has more max_series
        max_frames = self.max_frames
        # print(f'max_frames = {max_frames}')
        list_of_tensors_ = []
        for tensor in list_of_tensors:
            if self.config.args and not self.config.args.full_img_data:
                if tensor.shape[2] > max_frames // 2:
                    max_series = 1
            while tensor.shape[2] >= max_frames:  # downsample frames by 2 if too many
                tensor = F.avg_pool3d(tensor, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            # make tensor channel last
            tensor = rearrange(tensor, 'b c f h w -> b f h w c')
            list_of_tensors_.append(tensor)
        list_of_tensors = list_of_tensors_[:max_series]  # only use the first 10 series

        # list_of_tensors = [torch.randn(1, 256, 16, 16, 256, device=tensor.device)] * 1  # a test with dummy
        list_of_tensors_split = [split_video(tensor, self.max_frames, frame_axis=1) for tensor in list_of_tensors]
        # result in a list of tensor with variable length, each of shape (1, max_frame, h, w, c)
        list_of_tensors_split = [tensor for sublist in list_of_tensors_split for tensor in sublist]  # flatten the list

        all_features = []
        for key, linear in self.img_feature_linears.items():
            list_of_features = [linear(tensor) for tensor in list_of_tensors_split]  # change channels to hidden_size
            all_features += list_of_features

        all_cls = []
        attn_list = []
        for tensor in all_features:
            _, f, h, w, _ = tensor.shape
            tensor = rearrange(tensor, '1 f h w c -> (f h w) c')
            res_dict = self.gated_summarizer(tensor)
            cls = res_dict['cls']
            attn = res_dict['attention_raw']
            if self.move_attn_cpu:
                attn = attn.cpu().unsqueeze(dim=1)  # 1, 1, n_patches
                attn = rearrange(attn, '1 1 (f h w) -> 1 1 f h w', f=f, h=h, w=w)
            attn_list.append(attn)
            all_cls.append(cls)

        final_cls = sum(all_cls) / len(all_cls)
        final_cls = self.linear(final_cls)
        out = {
            'cls': final_cls,
            'attn_list': attn_list
        }
        return out


def get_gated_attnetion_video(config):
    dim_in = config.model.dim_video_former
    dim_out = config.model.dim_video_former
    dim_mid = 384
    gated_summarizer = GatedAttentionSummarizer(gate=True, dim_in=dim_in, dim_out=dim_out, dim_mid=dim_mid,
                                                dropout=True, n_classes=None)
    model = GatedVideoEventFormer(gated_summarizer=gated_summarizer, dim=dim_out, move_attn_cpu=config.model.move_attn_cpu,
                                  max_frames=config.model.max_video_frames, config=config)
    return model


class GatedTextSformer(nn.Module):
    def __init__(self, gated_summarizer, config):
        super(GatedTextSformer, self).__init__()
        self.gated_summarizer = gated_summarizer
        dim = config.hidden_size
        self.config = config
        # Linear layers for text features
        self.text_feature_linears = nn.ModuleDict({
            key: nn.Linear(FEATURE_DIMS[key], config.hidden_size)
            for key in config.txt_feature_keys
        })

    def forward(self, event):
        event_type = event['event_type']
        if event_type in ['影像检查', 'WSI']:
            raise ValueError(f'Event {event["event_type"]} on {event.get("date")} is not a text event.')

        processed_features = []
        all_attn = []
        # Process each text feature key
        for key in self.text_feature_linears:
            if 'txt_features' not in event or key not in event['txt_features'] or event['txt_features'][key] is None:
                raise ValueError(
                    f'Event {event["event_type"]} on {event.get("date")} has no text feature for key {key}\n{event}.')
            features = self.text_feature_linears[key](event['txt_features'][key])  # 1, L, dim
            L = features.shape[1]
            features = features.squeeze(0)  # L, dim
            res_dict = self.gated_summarizer(features)
            cls = res_dict['cls']  # 1, dim
            q_attn = res_dict['attention_raw']  # 1, split_len
            all_attn.append(q_attn.unsqueeze(1))  # 1, h, L
            processed_features.append(cls)

        # Fuse text features
        fused_feature = sum(processed_features) / len(processed_features)

        res = {
            'cls': fused_feature,
            'attn_list': all_attn
        }
        return res


def get_gated_attention_text(config):
    dim_in = config.hidden_size
    dim_out = config.hidden_size
    dim_mid = 512
    gated_summarizer = GatedAttentionSummarizer(gate=True, dim_in=dim_in, dim_out=dim_out, dim_mid=dim_mid,
                                                dropout=True, n_classes=None)
    model = GatedTextSformer(gated_summarizer, config)
    return model

