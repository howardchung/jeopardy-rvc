#!/usr/bin/env python3
"""
Persistent local RVC v2 + Edge-TTS voice generation worker for Jeopardy!.
Uses the AlexTrebek.zip weights (AlexTrebek_350e.pth + FAISS index) and
communicates with the Node.js server via newline-delimited JSON on stdin/stdout.
"""
from __future__ import annotations

import asyncio
import fcntl
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import warnings
import zipfile
from itertools import chain
from pathlib import Path

# Keep real stdout exclusively for JSON protocol messages; send all library output to stderr
_protocol_out = sys.stdout
sys.stdout = sys.stderr

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import edge_tts
import faiss
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel
from scipy import signal
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from transformers import HubertConfig, HubertModel

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
TREBEK_ZIP = ROOT_DIR / "AlexTrebek.zip"
TREBEK_ZIP_PARTS = [
    ROOT_DIR / "AlexTrebek.zip.part1",
    ROOT_DIR / "AlexTrebek.zip.part2",
]
TREBEK_DIR = MODELS_DIR / "AlexTrebek"
TREBEK_PTH = TREBEK_DIR / "AlexTrebek_350e.pth"
TREBEK_INDEX = TREBEK_DIR / "added_IVF1177_Flat_nprobe_1_v2.index"
RMVPE_PATH = MODELS_DIR / "predictors" / "rmvpe.pt"
CONTENTVEC_DIR = MODELS_DIR / "embedders" / "contentvec"

HF_RESOURCES_BASE = (
    "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/Resources"
)

# High-pass filter constants (16kHz input)
FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48
SAMPLE_RATE = 16000
bh, ah = signal.butter(
    N=FILTER_ORDER,
    Wn=CUTOFF_FREQUENCY,
    btype="high",
    fs=SAMPLE_RATE,
)

LRELU_SLOPE = 0.1
N_MELS = 128
N_CLASS = 360


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    return [item for sublist in l for item in sublist]


def sequence_mask(length: torch.Tensor, max_length: int | None = None):
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
        heads_share: bool = True,
        block_length: int | None = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        x, _ = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size:
            scores += self._compute_relative_scores(query, t_s)
        if self.proximal_bias:
            scores += self._attention_bias_proximal(t_s).to(scores.device, scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length:
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)

        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)
        if self.window_size:
            output += self._apply_relative_values(p_attn, t_s)
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _compute_relative_scores(self, query, length):
        rel_emb = self._get_relative_embeddings(self.emb_rel_k, length)
        rel_logits = torch.matmul(
            query / math.sqrt(self.k_channels), rel_emb.unsqueeze(0).transpose(-2, -1)
        )
        return self._relative_position_to_absolute_position(rel_logits)

    def _apply_relative_values(self, p_attn, length):
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        rel_emb = self._get_relative_embeddings(self.emb_rel_v, length)
        return torch.matmul(rel_weights, rel_emb.unsqueeze(0))

    def _get_relative_embeddings(self, embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            embeddings = F.pad(
                embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        return embeddings[:, start:end]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view(batch, heads, length * 2 * length)
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view(batch, heads, length + 1, 2 * length - 1)[
            :, :, :length, length - 1 :
        ]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view(batch, heads, length**2 + length * (length - 1))
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view(batch, heads, length, 2 * length)[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = r.unsqueeze(0) - r.unsqueeze(1)
        return -torch.log1p(torch.abs(diff)).unsqueeze(0).unsqueeze(0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str | None = None,
        causal: bool = False,
    ):
        super().__init__()
        self.padding_fn = self._causal_padding if causal else self._same_padding
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)
        self.activation = activation

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding_fn(x * x_mask))
        x = x * torch.sigmoid(1.702 * x) if self.activation == "gelu" else torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        pad_l, pad_r = self.conv_1.kernel_size[0] - 1, 0
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))

    def _same_padding(self, x):
        pad = (self.conv_1.kernel_size[0] - 1) // 2
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad, pad]]))


class WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.n_channels_tensor = torch.IntTensor([hidden_channels])

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1),
                name="weight",
            )

        dilations = [dilation_rate**i for i in range(n_layers)]
        paddings = [(kernel_size * d - d) // 2 for d in dilations]

        for i in range(n_layers):
            self.in_layers.append(
                weight_norm(
                    nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilations[i],
                        padding=paddings[i],
                    ),
                    name="weight",
                )
            )
            res_skip_channels = (
                hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            )
            self.res_skip_layers.append(
                weight_norm(
                    nn.Conv1d(hidden_channels, res_skip_channels, 1),
                    name="weight",
                )
            )

    def forward(self, x, x_mask, g=None):
        output = x.clone().zero_()
        g = self.cond_layer(g) if g is not None else None
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = (
                g[
                    :,
                    i * 2 * self.hidden_channels : (i + 1) * 2 * self.hidden_channels,
                    :,
                ]
                if g is not None
                else 0
            )
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_1 = nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_layers_2 = nn.ModuleList(
            [LayerNorm(hidden_channels) for _ in range(n_layers)]
        )

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        embedding_dim: int,
        f0: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.emb_phone = nn.Linear(embedding_dim, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = nn.Embedding(256, hidden_channels) if f0 else None
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor | None,
        lengths: torch.Tensor,
    ):
        x = self.emb_phone(phone)
        if pitch is not None and self.emb_pitch:
            x += self.emb_pitch(pitch)
        x *= math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = x.transpose(1, -1)
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations):
        layers = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilations
            ]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_residual = x
            x = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                x = x * x_mask
            x = F.leaky_relu(conv1(x), LRELU_SLOPE)
            if x_mask is not None:
                x = x * x_mask
            x = conv2(x)
            x = x + x_residual
        return x * x_mask if x_mask is not None else x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x


class SineGenerator(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        num_harmonics: int = 0,
        sine_amplitude: float = 0.1,
        noise_stddev: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_harmonics = num_harmonics
        self.sine_amplitude = sine_amplitude
        self.noise_stddev = noise_stddev
        self.voiced_threshold = voiced_threshold
        self.waveform_dim = self.num_harmonics + 1

    def _generate_sine_wave(self, f0: torch.Tensor, upsampling_factor: int):
        batch_size, length, _ = f0.shape
        upsampling_grid = torch.arange(
            1, upsampling_factor + 1, dtype=f0.dtype, device=f0.device
        )
        phase_increments = (f0 / self.sampling_rate) * upsampling_grid
        phase_remainder = torch.fmod(phase_increments[:, :-1, -1:] + 0.5, 1.0) - 0.5
        cumulative_phase = phase_remainder.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        phase_increments += F.pad(cumulative_phase, (0, 0, 1, 0), mode="constant")
        phase_increments = phase_increments.reshape(batch_size, -1, 1)
        harmonic_scale = torch.arange(
            1, self.waveform_dim + 1, dtype=f0.dtype, device=f0.device
        ).reshape(1, 1, -1)
        phase_increments *= harmonic_scale
        random_phase = torch.rand(1, 1, self.waveform_dim, device=f0.device)
        random_phase[..., 0] = 0
        phase_increments += random_phase
        return torch.sin(2 * np.pi * phase_increments)

    def forward(self, f0: torch.Tensor, upsampling_factor: int):
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = (
                self._generate_sine_wave(f0, upsampling_factor) * self.sine_amplitude
            )
            voiced_mask = (f0 > self.voiced_threshold).float()
            voiced_mask = F.interpolate(
                voiced_mask.transpose(2, 1),
                scale_factor=float(upsampling_factor),
                mode="nearest",
            ).transpose(2, 1)
            noise_amplitude = voiced_mask * self.noise_stddev + (1 - voiced_mask) * (
                self.sine_amplitude / 3
            )
            noise = noise_amplitude * torch.randn_like(sine_waves)
            sine_waveforms = sine_waves * voiced_mask + noise
        return sine_waveforms, voiced_mask, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ):
        super().__init__()
        self.l_sin_gen = SineGenerator(
            sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, upsample_factor: int = 1):
        sine_wavs, _, _ = self.l_sin_gen(x, upsample_factor)
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class HiFiGANNSFGenerator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate=sr, harmonic_num=0)
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            padding = (k - u) // 2 if u % 2 == 0 else u // 2 + u % 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        channels[i],
                        k,
                        u,
                        padding=padding,
                        output_padding=u % 2,
                    )
                )
            )
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        self.resblocks = nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )
        self.conv_post = nn.Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = math.prod(upsample_rates)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor | None = None
    ):
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = ups(x)
            x = x + noise_convs(har_source)
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / len(blocks)
        x = F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))


class Synthesizer(nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        use_f0: bool = True,
        text_enc_hidden_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.use_f0 = use_f0
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            text_enc_hidden_dim,
            f0=use_f0,
        )
        self.dec = HiFiGANNSFGenerator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor | None = None,
        nsff0: torch.Tensor | None = None,
        sid: torch.Tensor | None = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


# RMVPE pitch extractor classes
class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        return self.conv(x) + self.shortcut(x) if self.is_shortcut else self.conv(x) + x


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList(
            [ConvBlockRes(in_channels, out_channels, momentum)]
            + [
                ConvBlockRes(out_channels, out_channels, momentum)
                for _ in range(n_blocks - 1)
            ]
        )
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return (x, self.pool(x)) if self.kernel_size is not None else x


class RMVPEEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
            + [
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
                for _ in range(n_inters - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                (3, 3),
                stride,
                (1, 1),
                out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [ConvBlockRes(out_channels * 2, out_channels, momentum)]
            + [
                ConvBlockRes(out_channels, out_channels, momentum)
                for _ in range(n_blocks - 1)
            ]
        )

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class RMVPEDecoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.encoder = RMVPEEncoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = RMVPEDecoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, concat_tensors)


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sample_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa_mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp

    def forward(self, audio, center=True):
        key = str(audio.device)
        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(self.win_length).to(audio.device)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPE0Predictor:
    def __init__(self, model_path: str, device: str = "cpu"):
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()
        self.device = device
        self.model = model.to(device)
        self.mel_extractor = MelSpectrogram(
            N_MELS, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def infer_from_audio(self, audio: np.ndarray, thred: float = 0.03) -> np.ndarray:
        with torch.no_grad():
            audio_t = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
            mel = self.mel_extractor(audio_t, center=True)
            n_frames = mel.shape[-1]
            mel = F.pad(
                mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
            )
            hidden = self.model(mel)[:, :n_frames].squeeze(0).cpu().numpy()

        center = np.argmax(hidden, axis=1)
        salience = np.pad(hidden, ((0, 0), (4, 4)))
        center += 4
        starts = center - 4
        ends = center + 5
        todo_salience = np.array(
            [salience[idx, starts[idx] : ends[idx]] for idx in range(salience.shape[0])]
        )
        todo_cents = np.array(
            [self.cents_mapping[starts[idx] : ends[idx]] for idx in range(salience.shape[0])]
        )
        devided = np.sum(todo_salience * todo_cents, 1) / np.sum(todo_salience, 1)
        devided[np.max(salience, axis=1) <= thred] = 0
        f0 = 10 * (2 ** (devided / 1200))
        f0[f0 == 10] = 0
        return f0


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def download_if_missing(url: str, dest: Path):
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[rvc_worker] Downloading {url} -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MODELS_DIR / ".init.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not (TREBEK_PTH.is_file() and TREBEK_INDEX.is_file()):
                TREBEK_DIR.mkdir(parents=True, exist_ok=True)
                if all(p.is_file() for p in TREBEK_ZIP_PARTS):
                    print(
                        f"[rvc_worker] Extracting {', '.join(p.name for p in TREBEK_ZIP_PARTS)} -> {TREBEK_DIR}",
                        file=sys.stderr,
                    )
                    combined = io.BytesIO(
                        b"".join(p.read_bytes() for p in TREBEK_ZIP_PARTS)
                    )
                    with zipfile.ZipFile(combined, "r") as zf:
                        zf.extractall(TREBEK_DIR)
                elif TREBEK_ZIP.is_file():
                    print(
                        f"[rvc_worker] Extracting {TREBEK_ZIP} -> {TREBEK_DIR}",
                        file=sys.stderr,
                    )
                    with zipfile.ZipFile(TREBEK_ZIP, "r") as zf:
                        zf.extractall(TREBEK_DIR)
                else:
                    missing = [str(p) for p in TREBEK_ZIP_PARTS if not p.is_file()]
                    raise FileNotFoundError(
                        f"Missing AlexTrebek model archive parts: {', '.join(missing)}"
                    )

            download_if_missing(f"{HF_RESOURCES_BASE}/predictors/rmvpe.pt", RMVPE_PATH)
            download_if_missing(
                f"{HF_RESOURCES_BASE}/embedders/contentvec/config.json",
                CONTENTVEC_DIR / "config.json",
            )
            download_if_missing(
                f"{HF_RESOURCES_BASE}/embedders/contentvec/pytorch_model.bin",
                CONTENTVEC_DIR / "pytorch_model.bin",
            )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_hubert_contentvec(device: str) -> HubertModelWithFinalProj:
    config = HubertConfig.from_json_file(str(CONTENTVEC_DIR / "config.json"))
    model = HubertModelWithFinalProj(config)
    state_dict = torch.load(
        str(CONTENTVEC_DIR / "pytorch_model.bin"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).float()
    model.eval()
    return model


def change_rms(
    source_audio: np.ndarray,
    source_rate: int,
    target_audio: np.ndarray,
    target_rate: int,
    rate: float,
) -> np.ndarray:
    rms1 = librosa.feature.rms(
        y=source_audio,
        frame_length=source_rate // 2 * 2,
        hop_length=source_rate // 2,
    )
    rms2 = librosa.feature.rms(
        y=target_audio,
        frame_length=target_rate // 2 * 2,
        hop_length=target_rate // 2,
    )
    rms1_t = F.interpolate(
        torch.from_numpy(rms1).float().unsqueeze(0),
        size=target_audio.shape[0],
        mode="linear",
    ).squeeze()
    rms2_t = F.interpolate(
        torch.from_numpy(rms2).float().unsqueeze(0),
        size=target_audio.shape[0],
        mode="linear",
    ).squeeze()
    rms2_t = torch.maximum(rms2_t, torch.zeros_like(rms2_t) + 1e-6)
    return (
        target_audio
        * (torch.pow(rms1_t, 1 - rate) * torch.pow(rms2_t, rate - 1)).numpy()
    )


def process_audio_split(audio: np.ndarray, sr: int = 16000, silence_thresh: int = -60, min_silence_len: int = 250):
    frame_length = int(min_silence_len / 1000 * sr)
    hop_length = frame_length // 2
    intervals = librosa.effects.split(
        audio,
        top_db=-silence_thresh,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if len(intervals) == 0:
        return [audio], np.array([[0, len(audio)]])
    audio_segments = [audio[start:end] for start, end in intervals]
    return audio_segments, intervals


def merge_audio_segments(audio_segments_org, audio_segments_new, intervals, sr_orig, sr_new):
    merged_audio = np.array([], dtype=audio_segments_new[0].dtype)
    sr_ratio = sr_new / sr_orig
    for i, (start, end) in enumerate(intervals):
        start_new = int(start * sr_ratio)
        end_new = int(end * sr_ratio)
        original_duration = len(audio_segments_org[i]) / sr_orig
        new_duration = len(audio_segments_new[i]) / sr_new
        duration_diff = new_duration - original_duration
        silence_samples = int(abs(duration_diff) * sr_new)
        silence_compensation = np.zeros(
            silence_samples, dtype=audio_segments_new[0].dtype
        )
        if i == 0 and start_new > 0:
            initial_silence = np.zeros(start_new, dtype=audio_segments_new[0].dtype)
            merged_audio = np.concatenate((merged_audio, initial_silence))
        if duration_diff > 0:
            merged_audio = np.concatenate((merged_audio, silence_compensation))
        merged_audio = np.concatenate((merged_audio, audio_segments_new[i]))
        if duration_diff < 0:
            merged_audio = np.concatenate((merged_audio, silence_compensation))
        if i < len(intervals) - 1:
            next_start_new = int(intervals[i + 1][0] * sr_ratio)
            silence_duration = next_start_new - end_new
            if silence_duration > 0:
                silence = np.zeros(silence_duration, dtype=audio_segments_new[0].dtype)
                merged_audio = np.concatenate((merged_audio, silence))
    return merged_audio


def select_device_and_workers() -> tuple[str, int]:
    env_device = os.environ.get("RVC_DEVICE")
    env_workers = os.environ.get("RVC_NUM_WORKERS")
    worker_index = max(0, int(os.environ.get("RVC_WORKER_INDEX", "0")))

    if env_device:
        device = env_device
    elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu_idx = worker_index % torch.cuda.device_count()
        device = f"cuda:{gpu_idx}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if env_workers:
        recommended_workers = max(1, int(env_workers))
    elif device.startswith("cuda"):
        gpu_count = max(1, torch.cuda.device_count())
        try:
            vram_gb = (
                torch.cuda.get_device_properties(device).total_memory
                / (1024**3)
            )
            workers_per_gpu = 2 if vram_gb >= 8.0 else 1
        except Exception:
            workers_per_gpu = 1
        recommended_workers = gpu_count * workers_per_gpu
    elif device == "mps":
        recommended_workers = 2
    else:
        recommended_workers = max(1, os.cpu_count() or 1)

    return device, recommended_workers


class TrebekVoiceEngine:
    def __init__(self):
        threads = int(os.environ.get("RVC_THREADS_PER_WORKER", "1"))
        torch.set_num_threads(max(1, threads))
        ensure_models()
        self.device, self.recommended_workers = select_device_and_workers()
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        print(
            f"[rvc_worker] Initializing models on {self.device} (recommended workers: {self.recommended_workers})...",
            file=sys.stderr,
        )

        self.hubert = load_hubert_contentvec(self.device)
        self.rmvpe = RMVPE0Predictor(str(RMVPE_PATH), device=self.device)

        cpt = torch.load(str(TREBEK_PTH), map_location="cpu", weights_only=False)
        self.tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.use_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v2")
        text_enc_hidden_dim = 768 if self.version == "v2" else 256

        self.net_g = Synthesizer(
            *cpt["config"],
            use_f0=self.use_f0,
            text_enc_hidden_dim=text_enc_hidden_dim,
        )
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g = self.net_g.to(self.device).float()
        self.net_g.eval()

        # Preload FAISS index once in memory
        self.index = faiss.read_index(str(TREBEK_INDEX))
        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        self._configure_pipeline_windows()
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        print(f"[rvc_worker] Models loaded and ready on {self.device}.", file=sys.stderr)

    def _configure_pipeline_windows(self):
        # Adjust window sizes for low-VRAM GPUs (<= 4GB)
        if self.device.startswith("cuda"):
            try:
                gpu_mem = int(
                    torch.cuda.get_device_properties(self.device).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            except Exception:
                gpu_mem = 8
            if gpu_mem <= 4:
                self.x_pad, self.x_query, self.x_center, self.x_max = 1, 5, 30, 32
            else:
                self.x_pad, self.x_query, self.x_center, self.x_max = 1, 6, 38, 41
        else:
            self.x_pad, self.x_query, self.x_center, self.x_max = 1, 6, 38, 41

        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = self.tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max

    def _fallback_to_cpu(self):
        if self.device == "cpu":
            return
        print(
            f"[rvc_worker] GPU OOM on {self.device}; falling back to CPU...",
            file=sys.stderr,
        )
        self.device = "cpu"
        self.hubert = self.hubert.to("cpu")
        self.rmvpe.device = "cpu"
        self.rmvpe.model = self.rmvpe.model.to("cpu")
        self.rmvpe.mel_extractor = self.rmvpe.mel_extractor.to("cpu")
        self.net_g = self.net_g.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._configure_pipeline_windows()

    def _voice_conversion_segment(
        self,
        audio0: np.ndarray,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        index_rate: float = 0.5,
        protect: float = 0.33,
    ) -> np.ndarray:
        with torch.no_grad():
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            feats = feats.view(1, -1).to(self.device)
            feats = self.hubert(feats)["last_hidden_state"]
            if self.version == "v1":
                feats = self.hubert.final_proj(feats[0]).unsqueeze(0)

            feats0 = feats.clone()
            if self.index is not None and index_rate > 0:
                npy = feats[0].cpu().numpy()
                score, ix = self.index.search(npy, k=8)
                weight = np.square(1.0 / np.maximum(score, 1e-6))
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats = (
                    torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                    + (1 - index_rate) * feats
                )

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])

            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            pitch_seg, pitchf_seg = pitch[:, :p_len], pitchf[:, :p_len]
            if protect < 0.5:
                pitchff = pitchf_seg.clone()
                pitchff[pitchf_seg > 0] = 1
                pitchff[pitchf_seg < 1] = protect
                feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                    1 - pitchff.unsqueeze(-1)
                )
                feats = feats.to(feats0.dtype)

            p_len_t = torch.tensor([p_len], device=self.device).long()
            sid = torch.tensor([0], device=self.device).long()
            audio1 = (
                self.net_g.infer(
                    feats.float(), p_len_t, pitch_seg, pitchf_seg.float(), sid
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            return audio1

    def _convert_chunk(
        self,
        audio: np.ndarray,
        pitch_shift: int = 0,
        index_rate: float = 0.5,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> np.ndarray:
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        f0 = self.rmvpe.infer_from_audio(audio_pad, thred=0.03)[1:]
        f0 *= pow(2, pitch_shift / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        pitch = torch.tensor(f0_coarse[:p_len], device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(f0bak[:p_len], device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(
                self._voice_conversion_segment(
                    audio_pad[s : t + self.t_pad2 + self.window],
                    pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                    pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                    index_rate=index_rate,
                    protect=protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t

        audio_opt.append(
            self._voice_conversion_segment(
                audio_pad[t:],
                pitch[:, t // self.window :] if t is not None else pitch,
                pitchf[:, t // self.window :] if t is not None else pitchf,
                index_rate=index_rate,
                protect=protect,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(
                audio, self.sample_rate, audio_opt, self.tgt_sr, rms_mix_rate
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
        return audio_opt

    def synthesize_to_mp3(
        self,
        text: str,
        output_path: str,
        voice: str = "en-US-ChristopherNeural",
        split_audio: bool = True,
        clean_audio: bool = True,
        clean_strength: float = 0.7,
        output_sr: int = 44100,
    ) -> str:
        out = Path(output_path)
        if out.is_file() and out.stat().st_size > 0:
            return str(out)

        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tts_mp3 = Path(tmpdir) / "tts.mp3"
            tts_wav = Path(tmpdir) / "tts_16k.wav"
            rvc_wav = Path(tmpdir) / "rvc.wav"
            tmp_out = out.with_suffix(f".{os.getpid()}.tmp.mp3")

            # 1. Edge TTS
            communicate = edge_tts.Communicate(
                text, voice, pitch="+0Hz", rate="+0%", volume="+0%"
            )
            asyncio.run(communicate.save(str(tts_mp3)))

            # 2. Decode to 16kHz mono wav via ffmpeg
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tts_mp3),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    str(tts_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            audio, _ = sf.read(str(tts_wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            # 3. RVC conversion (with optional segment splitting)
            if split_audio:
                chunks, intervals = process_audio_split(audio, 16000)
            else:
                chunks = [audio]
                intervals = np.array([[0, len(audio)]])

            try:
                converted_chunks = [
                    self._convert_chunk(c) for c in chunks if len(c) > 0
                ]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device != "cpu":
                    self._fallback_to_cpu()
                    converted_chunks = [
                        self._convert_chunk(c) for c in chunks if len(c) > 0
                    ]
                else:
                    raise

            if split_audio and len(converted_chunks) > 1:
                audio_opt = merge_audio_segments(
                    chunks, converted_chunks, intervals, 16000, self.tgt_sr
                )
            else:
                audio_opt = converted_chunks[0]

            # 4. Optional noise reduction
            if clean_audio:
                try:
                    use_gpu_nr = self.device.startswith("cuda")
                    cleaned = nr.reduce_noise(
                        y=audio_opt,
                        sr=self.tgt_sr,
                        prop_decrease=clean_strength,
                        use_torch=use_gpu_nr,
                        device=self.device if use_gpu_nr else None,
                    )
                    if cleaned is not None:
                        audio_opt = cleaned
                except Exception as e:
                    print(f"[rvc_worker] Noise reduction warning: {e}", file=sys.stderr)

            sf.write(str(rvc_wav), audio_opt, self.tgt_sr, format="WAV")

            # 5. Encode final MP3 at output_sr atomically
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(rvc_wav),
                    "-ar",
                    str(output_sr),
                    "-codec:a",
                    "libmp3lame",
                    "-qscale:a",
                    "2",
                    str(tmp_out),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tmp_out.replace(out)

        return str(out)


def send_msg(obj: dict):
    _protocol_out.write(json.dumps(obj) + "\n")
    _protocol_out.flush()


def main():
    try:
        engine = TrebekVoiceEngine()
        send_msg(
            {
                "type": "ready",
                "device": engine.device,
                "recommended_workers": engine.recommended_workers,
            }
        )
    except Exception as e:
        send_msg({"type": "init_error", "error": str(e), "trace": traceback.format_exc()})
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            text = req.get("text", "")
            output_path = req.get("output_path", "")
            start = time.time()
            path = engine.synthesize_to_mp3(text=text, output_path=output_path)
            elapsed_ms = int((time.time() - start) * 1000)
            send_msg({"id": req_id, "ok": True, "path": path, "elapsed_ms": elapsed_ms})
        except Exception as e:
            send_msg(
                {
                    "id": req_id,
                    "ok": False,
                    "error": str(e),
                    "trace": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
