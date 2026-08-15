"""Unit tests for 1D time-series neural network architectures (InceptionTime, ResNet1D, AttentiveLSTM)."""

import pytest
import torch
from src.models.timeseries_models import AttentiveLSTM, InceptionTime, ResNet1D


def test_inception_time_forward_pass():
    batch_size = 4
    n_sensors = 6
    seq_len = 300
    n_classes = 50

    x = torch.randn(batch_size, n_sensors, seq_len)
    model = InceptionTime(n_sensors=n_sensors, n_classes=n_classes, num_blocks=2, out_channels=16)
    out = model(x)

    assert out.shape == (batch_size, n_classes)
    assert not torch.isnan(out).any()


def test_inception_time_channel_last_format():
    batch_size = 3
    n_sensors = 8
    seq_len = 200
    n_classes = 20

    # Shape: (batch_size, seq_len, n_sensors)
    x = torch.randn(batch_size, seq_len, n_sensors)
    model = InceptionTime(n_sensors=n_sensors, n_classes=n_classes, num_blocks=1, out_channels=16)
    out = model(x)

    assert out.shape == (batch_size, n_classes)


def test_resnet1d_forward_pass():
    batch_size = 2
    n_sensors = 6
    seq_len = 300
    n_classes = 15

    x = torch.randn(batch_size, n_sensors, seq_len)
    model = ResNet1D(n_sensors=n_sensors, n_classes=n_classes, layer_sizes=(32, 64))
    out = model(x)

    assert out.shape == (batch_size, n_classes)


def test_attentive_lstm_forward_pass():
    batch_size = 3
    n_sensors = 6
    seq_len = 150
    n_classes = 10

    x = torch.randn(batch_size, seq_len, n_sensors)
    model = AttentiveLSTM(n_sensors=n_sensors, n_classes=n_classes, hidden_size=32, num_layers=1)
    out = model(x)

    assert out.shape == (batch_size, n_classes)
