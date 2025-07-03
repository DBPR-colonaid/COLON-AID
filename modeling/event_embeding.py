import torch
import torch.nn as nn
import einops
from modeling.constants import FEATURE_DIMS_V3


class AbstractEventEmbedding(nn.Module):
    """
    Abstract class for event embedding.
    After embedding, event will have a new attribute 'embedding', storing the embedding features.
    """

    def __init__(self, config, img_feature_keys=None, txt_feature_keys=None, hidden_size=None, legecy_mode=True):
        super(AbstractEventEmbedding, self).__init__()
        self.img_feature_keys = config.img_feature_keys
        self.txt_feature_keys = config.txt_feature_keys
        self.hidden_size = config.hidden_size
        self.device = config.device
        self.legacy_mode = legecy_mode

        if img_feature_keys is not None:
            self.img_feature_keys = img_feature_keys
            print(f'Overriding image feature keys by the input: {img_feature_keys}')
        if txt_feature_keys is not None:
            self.txt_feature_keys = txt_feature_keys
            print(f'Overriding text feature keys by the input: {txt_feature_keys}')
        if hidden_size is not None:
            self.hidden_size = hidden_size
            print(f'Overriding hidden size by the input: {hidden_size}')

        self.config = config

    def embed_img_event(self, event):
        raise NotImplementedError

    def embed_text_event(self, event):
        raise NotImplementedError

    def embed_lab_event(self, event):
        raise NotImplementedError

    def forward(self, event):
        if 'embedding' in event:  # already embedded
            print(f'Event {event["event_type"]} on {event.get("date", "birth_date")} already embedded.')
            return event['embedding']

        event_type = event['event_type']
        if event_type == '影像检查':
            embd = self.embed_img_event(event)
        elif event_type == '化验':
            embd = self.embed_lab_event(event)
        else:
            embd = self.embed_text_event(event)
        if embd is None:
            embd = torch.zeros(1, self.hidden_size, device=self.device)  # (1, d)
        if self.legacy_mode:
            event['embedding'] = embd
            return embd
        res_dict = {'cls': embd, 'attn_list': []}
        return res_dict


class AvgEventEmbedding(AbstractEventEmbedding):
    def __init__(self, config, img_feature_keys=None, txt_feature_keys=None, hidden_size=None, legacy_mode=True):
        super(AvgEventEmbedding, self).__init__(config, img_feature_keys, txt_feature_keys, hidden_size, legacy_mode)
        fd = FEATURE_DIMS_V3
        for key in set(self.img_feature_keys):
            setattr(self, f'{key}_linear', nn.Linear(fd[key], self.hidden_size))

        for key in self.txt_feature_keys:
            setattr(self, f'{key}_linear', nn.Linear(fd[key], self.hidden_size))

        self.lab_linear = nn.Linear(fd['lab'], self.hidden_size)

    def embed_img_event(self, event):
        features = []
        if 'img_features' not in event or not event['img_features']:
            # import pdb; pdb.set_trace()
            try:
                self.config.logger.debug(f'Event {event["event_type"]} in {event["content"]} {event.get("pacs", "PACS-None")}-{event["uid"]} on {event.get("date", "birth_date")} has no image features.')
            except KeyError as e:
                self.config.logger.debug(f'Event {event} has no image features.')
            return torch.zeros(1, self.hidden_size, device=self.device, requires_grad=True)
        assert isinstance(event['img_features'], dict)
        for key in self.img_feature_keys:
            ff = 0
            if key not in event['img_features'] or not event['img_features'][key]:
                continue  # some events may not have all feature types
            for event_feature in event['img_features'][key]:
                event_feature = event_feature['features']
                if len(event_feature.shape) == 5:
                    ff += torch.mean(event_feature, dim=[0, 2, 3, 4])  # (num, 3072, 4, 4, 4) -> (3072,)
                else:
                    ff += torch.mean(event_feature, dim=0)  # (num, 4096) -> (4096,)
            ff /= len(event['img_features'][key])
            ff = getattr(self, f'{key}_linear')(ff)  # (d,)
            features.append(ff)
        if features:
            # features = torch.stack(features, dim=0)  # (num_img_features_keys, d)
            features = sum(features) / len(features)  # (d,)
            features = features[None]  # (1, d)
        else:
            print(f'Event {event["pacs"]} {event["content"]} {event["uid"]} {event["event_type"]} on {event.get("date", "birth_date")} has [empty] image features.')
            features = torch.zeros(1, self.hidden_size, device=self.device, requires_grad=True)
        return features

    def embed_text_event(self, event):
        features = []
        if 'txt_features' not in event or not event['txt_features']:
            print(f'Event {event["event_type"]} on {event.get("date", "birth_date")} has no text features.')
            return torch.zeros(1, self.hidden_size, device=self.device, requires_grad=True)
        assert isinstance(event['txt_features'], dict)
        for key in self.txt_feature_keys:
            event_feature = event['txt_features'][key]
            if len(event_feature.shape) == 3:
                event_feature = event_feature.squeeze(dim=0)  # (1, L, d) -> (L, d)
            event_feature = getattr(self, f'{key}_linear')(event_feature)  # (L, d_model)
            features.append(event_feature.mean(dim=0))  # (d_model,) # average over the sequence
        # features = torch.stack(features, dim=0)  # (num_txt_features_keys, d)
        if features:
            features = sum(features) / len(features)  # (d,)
            features = features[None]  # (1, d)
        else:
            print(f'Event {event["event_type"]} on {event.get("date", "birth_date")} has no text features.')
            return torch.zeros(1, self.hidden_size, device=self.device, requires_grad=True)
        return features

    def embed_lab_event(self, event):
        if 'lab_features' not in event:
            print(f'Event {event["event_type"]} on {event.get("date", "birth_date")} has no lab features.')
            return torch.zeros(1, self.hidden_size, device=self.device, requires_grad=True)
        f = event['lab_features']
        f = self.lab_linear(f.reshape(1, FEATURE_DIMS_V3['lab']))  # (1, d)
        return f
