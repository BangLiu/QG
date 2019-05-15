import torch
import torch.nn as nn
from .modules.cnn import Initialized_Conv1d
from .modules.highway import Highway
from .modules.gcn import GCN


class CluePredictor_gcn(nn.Module):
    def __init__(self, config, total_emb_dim, PAD=0, dropout=0.1):
        super().__init__()
        self.config = config
        self.PAD = PAD
        self.dropout = dropout
        # transform embedding
        self.conv1d = Initialized_Conv1d(
            total_emb_dim, config.gcn_hidden_size, bias=False)
        self.high = Highway(2, config.gcn_hidden_size)
        # encoding
        self.gcn = GCN(
            config.gcn_hidden_size,
            config.gcn_hidden_size,
            config.gcn_num_layers, dropout=dropout)
        # is clue word classification
        self.conv = Initialized_Conv1d(
            config.gcn_num_layers * config.gcn_hidden_size, 1,
            relu=True, bias=True)
        self.linear = nn.Linear(config.sent_limit, config.sent_limit)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, batch, Semb):
        # shape of Semb: batch_size * hidden_dim * seq_len
        device = Semb.get_device()
        batch_size, emb_dim, seq_length = Semb.shape

        # get sentence embedding
        Semb = self.conv1d(Semb)
        Semb = self.high(Semb)
        maskS = (torch.ones_like(batch["ans_sent_word_ids"]) *
                 self.PAD != batch["ans_sent_word_ids"]).float()

        # get answer embedding
        start, end = batch["y1_in_sent"], batch["y2_in_sent"]
        max_ans_len = int((end - start + 1).max().item())
        Aemb = torch.zeros(
            [batch_size, self.config.d_model, max_ans_len]).to(device)
        maskA = torch.zeros(
            [batch_size, max_ans_len]).to(device)
        for i in range(batch_size):
            # get predicted answer sentence encoding
            ans_emb = Semb[i, :, start[i]:end[i] + 1]
            La = ans_emb.shape[1]  # because ans_emb is an 2D tensor, not 3D
            Aemb[i, :, :La] = ans_emb
            maskA[i, :La] = 1.0

        # get y_clue
        adj = torch.zeros([batch_size, seq_length, seq_length]).to(device)
        for i in range(batch_size):
            edges = batch["ans_sent_syntactic_edges"][i]
            for e in edges:
                src_idx, tgt_idx = e[0], e[1]
                adj[i, src_idx, tgt_idx] = 1.0
                if not self.config.gcn_directed:
                    adj[i, tgt_idx, src_idx] = 1.0
        gcn_outputs, mask = self.gcn(adj, Semb.transpose(1, 2))
        y_clue_logits = self.linear(self.conv(
            torch.cat(gcn_outputs, 2).transpose(1, 2))).squeeze()
        y_clue_logits = y_clue_logits * maskS + (1 - maskS) * (-1e30)

        return y_clue_logits, maskA
