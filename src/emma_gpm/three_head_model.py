"""Frozen multilevel VideoMAE probes and explicit sequential transfer controls."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import VideoMAEModel

from .ssl_model import encoder_source_config, load_encoder_weights
from .videomae import inflate_rgb_patch_embedding
from .three_head_data import CLASS_NAMES


def make_encoder(source, kind='pretrained'):
    config = encoder_source_config(source, SimpleNamespace(history_frames=16))
    if kind == 'rgb':
        checkpoint = 'MCG-NJU/videomae-small-finetuned-kinetics'
        rgb = VideoMAEModel.from_pretrained(checkpoint)
        if rgb.config.hidden_size != config.hidden_size or rgb.config.num_hidden_layers != 12:
            raise ValueError('RGB control must match the SEVIRI encoder architecture')
        # Retain the source LayerNorm/activation settings as part of this control.
        config = deepcopy(rgb.config)
        config.num_channels, config.num_frames, config.norm_pix_loss = 7, 16, False
        encoder = VideoMAEModel(config)
        weights = rgb.state_dict()
        key = 'embeddings.patch_embeddings.projection.weight'
        weights[key] = inflate_rgb_patch_embedding(weights[key], 7)
        encoder.load_state_dict(weights, strict=True)
        provenance = dict(kind=kind, source=checkpoint,
            source_training='Natural-video MAE pretraining followed by supervised Kinetics fine-tuning',
            input_adaptation='Mean RGB projection replicated to seven channels, scaled by 3/7')
    else:
        if kind not in ('pretrained', 'random'):
            raise ValueError('Unknown encoder initialization')
        encoder = VideoMAEModel(config)
        provenance = load_encoder_weights(encoder, source) if kind == 'pretrained' else {'kind': kind}
    provenance.update(probe_frames=16, positional_adaptation='Rebuilt fixed sinusoidal positions for 16 frames')
    return encoder, provenance


class LightweightFPN3D(nn.Module):
    """Four lateral projections and progressive depthwise 3D refinement."""
    def __init__(self, dimension, width=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv3d(dimension, width, 1) for _ in range(4)])
        self.refine = nn.ModuleList([nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1, groups=width),
            nn.GroupNorm(8, width), nn.GELU()) for _ in range(4)])

    def forward(self, features, output_size=None):
        if len(features) != 4:
            raise ValueError('FPN requires encoder blocks 3, 6, 9 and 12')
        x = None
        for stage, index in enumerate((3, 2, 1, 0)):
            lateral = self.lateral[index](features[index])
            size = (lateral.shape[2], lateral.shape[3] * 2**stage, lateral.shape[4] * 2**stage)
            lateral = F.interpolate(lateral, size=size, mode='trilinear', align_corners=False)
            x = lateral if x is None else lateral + F.interpolate(x, size=size, mode='trilinear', align_corners=False)
            x = self.refine[index](x)
        return x if output_size is None else F.interpolate(x, size=output_size, mode='trilinear', align_corners=False)


class ThreeHeadProbe(nn.Module):
    def __init__(self, encoder, heads=('cma', 'ct', 'rdt'), width=128, encoder_mode='frozen'):
        super().__init__()
        if not heads or any(head not in CLASS_NAMES for head in heads):
            raise ValueError('Unknown or empty probe tasks')
        if encoder.config.num_hidden_layers != 12:
            raise ValueError('Multilevel protocol requires a twelve-block encoder')
        self.encoder, self.head_names = encoder, tuple(heads)
        self.fpn = LightweightFPN3D(encoder.config.hidden_size, width)
        self.heads = nn.ModuleDict()
        for name in self.head_names:
            conditioning = int(name in ('ct', 'rdt') and 'cma' in self.head_names)
            conditioning += 5 * int(name == 'rdt' and 'ct' in self.head_names)
            self.heads[name] = nn.Conv3d(width + conditioning, 1 if name == 'cma' else 5, 1)
        self.auxiliary_mae = None
        self.set_encoder_mode(encoder_mode)

    def set_encoder_mode(self, mode):
        if mode not in ('frozen', 'last4', 'full'):
            raise ValueError('Encoder mode must be frozen, last4 or full')
        self.encoder_mode = mode
        self.encoder.requires_grad_(mode == 'full')
        if mode == 'last4':
            for block in self.encoder.encoder.layer[-4:]:
                block.requires_grad_(True)
            if getattr(self.encoder, 'layernorm', None) is not None:
                self.encoder.layernorm.requires_grad_(True)
        if mode == 'frozen':
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_mode == 'frozen':
            self.encoder.eval()
        return self

    def encode(self, clips):
        cfg = self.encoder.config
        if tuple(clips.shape[1:]) != (cfg.num_frames, 7, cfg.image_size, cfg.image_size):
            raise ValueError('Probe expects complete canonical sixteen-frame seven-channel clips')
        context = torch.no_grad() if self.encoder_mode == 'frozen' else nullcontext()
        with context:
            states = self.encoder(clips, output_hidden_states=True).hidden_states
        t, grid = cfg.num_frames // cfg.tubelet_size, cfg.image_size // cfg.patch_size
        return [states[i].reshape(len(clips), t, grid, grid, cfg.hidden_size)
                .permute(0, 4, 1, 2, 3).contiguous() for i in (3, 6, 9, 12)]

    def decode(self, features):
        cfg = self.encoder.config
        shared = self.fpn(features)
        size = (cfg.num_frames, cfg.image_size, cfg.image_size)
        output = {}
        for head in ('cma', 'ct', 'rdt'):
            if head not in self.head_names:
                continue
            # A pointwise linear projection commutes with trilinear upsampling.
            # Evaluate that projection on the coarse shared tensor, avoiding
            # the otherwise enormous 128-channel full-resolution allocation.
            # Nonlinear conditioning still uses full-resolution probabilities.
            layer = self.heads[head]
            channels = shared.shape[1]
            projected = F.conv3d(shared, layer.weight[:, :channels], layer.bias)
            output[head] = F.interpolate(projected, size=size, mode='trilinear', align_corners=False)
            parts = []
            if head in ('ct', 'rdt') and 'cma' in output:
                parts.append(output['cma'].sigmoid())
            if head == 'rdt' and 'ct' in output:
                parts.append(output['ct'].softmax(1))
            if parts:
                output[head] = output[head] + F.conv3d(torch.cat(parts, dim=1), layer.weight[:, channels:])
        return output

    def forward(self, clips=None, *, features=None):
        return self.decode(self.encode(clips) if features is None else features)


def class_weights(counts):
    """Inverse square-root training support, normalized over present classes."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    weights = torch.where(counts > 0, counts.clamp_min(1).rsqrt(), 0.)
    if (counts > 0).any():
        weights /= weights[counts > 0].mean()
    return weights


def segmentation_loss(logits, labels, *, class_weight=None, gamma=2., generalized=False):
    """Focal CE and Dice, with every 255-valued pixel excluded from both terms."""
    logits = logits.float()
    valid = labels != 255
    if not valid.any():
        return logits.sum() * 0.
    if ((labels[valid] < 0) | (labels[valid] >= logits.shape[1])).any():
        raise ValueError('Target outside class range')
    safe = labels.long().masked_fill(~valid, 0)
    log_probability = F.log_softmax(logits, 1)
    selected = log_probability.gather(1, safe.unsqueeze(1)).squeeze(1)
    focal = -(1.-selected.exp()).pow(gamma) * selected
    if class_weight is not None:
        focal = focal * class_weight.to(logits)[safe]
    focal = focal[valid].mean()
    probabilities = log_probability.exp() * valid.unsqueeze(1)
    truth = F.one_hot(safe, logits.shape[1]).movedim(-1, 1).float() * valid.unsqueeze(1)
    dims = (0, 2, 3, 4)
    intersection = (probabilities * truth).sum(dims)
    support = truth.sum(dims)
    denominator = probabilities.sum(dims) + support
    # Absent classes have no estimable Dice recall and are handled by focal CE.
    present = support > 0
    if generalized:
        weight = torch.where(present, support.clamp_min(1).square().reciprocal(), 0.)
        weight = weight / weight.sum()
        dice = 1. - (2.*(weight*intersection).sum()+1e-6)/((weight*denominator).sum()+1e-6)
    else:
        dice = 1. - ((2.*intersection[present]+1e-6)/(denominator[present]+1e-6)).mean()
    return focal + dice


def probe_loss(outputs, targets, *, rdt_class_weights=None, nesting_weight=.1):
    terms = {}
    for head, logits in outputs.items():
        labels = targets[head]
        if head == 'cma':
            valid = labels != 255
            z = logits[:, 0].float()
            if not valid.any():
                terms[head] = z.sum()*0.
                continue
            if not torch.isin(labels[valid], torch.tensor([0, 1], device=labels.device)).all():
                raise ValueError('Invalid CMA label')
            truth = labels[valid].float()
            probability = z[valid].sigmoid()
            dice = 1. - (2.*(probability*truth).sum()+1e-6)/(probability.sum()+truth.sum()+1e-6)
            terms[head] = F.binary_cross_entropy_with_logits(z[valid], truth) + dice
        else:
            terms[head] = segmentation_loss(logits, labels,
                class_weight=rdt_class_weights if head == 'rdt' else None, generalized=head == 'rdt')
    nesting = next(iter(outputs.values())).float().sum()*0.
    if 'ct' in outputs:
        p_ct = outputs['ct'].float().softmax(1)
        if 'cma' in outputs:
            valid = (targets['ct'] != 255) & (targets['cma'] != 255)
            if valid.any():
                nesting = nesting + F.relu(1.-p_ct[:, 0]-outputs['cma'][:, 0].float().sigmoid())[valid].square().mean()
        if 'rdt' in outputs:
            valid = (targets['ct'] != 255) & (targets['rdt'] != 255)
            if valid.any():
                p_conv = outputs['rdt'].float().softmax(1)[:, 2:].sum(1)
                nesting = nesting + F.relu(p_conv-p_ct[:, 3])[valid].square().mean()
    terms['nesting'] = nesting
    return sum(value for key, value in terms.items() if key != 'nesting') + nesting_weight*nesting, terms


class SegmentationMetrics:
    """Pixel-pooled confusion and multiclass Brier; absent support stays explicit."""
    def __init__(self, classes):
        self.classes = classes
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.brier_sum = 0.
        self.count = 0
        self.binary = np.zeros((2, 2), dtype=np.int64) if classes == 5 else None
        self.binary_brier_sum = 0.

    @torch.no_grad()
    def update(self, logits, labels, *, convective=False):
        valid = labels != 255
        truth = labels[valid].long()
        probabilities = logits.float().sigmoid() if self.classes == 2 else logits.float().softmax(1)
        if self.classes == 2:
            probabilities = torch.cat((1.-probabilities, probabilities), 1)
        p = probabilities.movedim(1, -1)[valid]
        if len(truth) == 0:
            return
        prediction = p.argmax(-1)
        counts = torch.bincount(self.classes*truth+prediction, minlength=self.classes**2)
        self.confusion += counts.reshape(self.classes, self.classes).cpu().numpy()
        self.brier_sum += float((p.square().sum(-1)-2.*p.gather(1, truth[:, None]).squeeze(1)+1.).double().sum())
        self.count += len(truth)
        if convective:
            pc = p[:, 2:].sum(-1)
            tc = truth >= 2
            self.binary += torch.bincount(2*tc.long()+(pc >= .5).long(), minlength=4).reshape(2,2).cpu().numpy()
            self.binary_brier_sum += float((pc-tc.float()).double().square().sum())

    @staticmethod
    def scores(matrix):
        tp = np.diag(matrix)
        true, predicted = matrix.sum(1), matrix.sum(0)
        union = true+predicted-tp
        def divided(a, b):
            return [float(x/y) if y else None for x, y in zip(a, b)]
        return dict(support=true.tolist(), predicted=predicted.tolist(),
                    iou=divided(tp, union), csi=divided(tp, union),
                    f1=divided(2*tp, true+predicted), confusion=matrix.tolist())

    def result(self, *, convective=False):
        result = {**self.scores(self.confusion), 'pixels': self.count,
                  'brier_multiclass': self.brier_sum/self.count if self.count else None}
        if convective:
            result['active_convection'] = {**self.scores(self.binary),
                'brier': self.binary_brier_sum/self.count if self.count else None}
        return result


def diagnostic_input(clips, mode, *, seed=0):
    """Evaluation-only interventions; ordinary training preserves all observations."""
    result = clips.clone()
    if mode == 'all_channels':
        return result
    if mode == 'shuffle_time':
        generator = torch.Generator(device=clips.device).manual_seed(seed)
        order = torch.randperm(clips.shape[1], generator=generator, device=clips.device)
        return result[:, order]
    if mode == 'repeat_last':
        return result[:, -1:].expand_as(result).contiguous()
    keep = {'teacher_two_channels': (0, 4), 'wv_and_ir108': (0, 1, 4),
            'without_ir134': (0, 1, 2, 3, 4, 5)}
    if mode not in keep:
        raise ValueError('Unknown diagnostic intervention')
    # Zero in globally normalized space means the shared training mean Kelvin.
    for channel in set(range(7))-set(keep[mode]):
        result[:, :, channel] = 0.
    return result
