import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trim = pad
        self.residual = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        return F.gelu(out + self.residual(x))

class DrowsinessNet(nn.Module):
    def __init__(self, feat_dim=20, seq_len=16, num_classes=2,
                 cnn_channels=None, lstm_hidden=256,
                 lstm_layers=2, fc_hidden=128, dropout=0.4):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        layers = []
        in_ch = feat_dim
        for i, out_ch in enumerate(cnn_channels):
            layers.append(TemporalBlock(in_ch, out_ch,
                                        dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fc_hidden // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)
        return self.classifier(context)
