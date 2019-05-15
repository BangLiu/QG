import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.cnn import Initialized_Conv1d, DepthwiseSeparableConv
from .modules.position import PosEncoder
from .modules.highway import Highway


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1d(
            in_channels=d_model, out_channels=d_model * 2,
            kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries
        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head)
                for tensor in torch.split(memory, self.d_model, dim=2)]
        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x is not None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [DepthwiseSeparableConv(d_model, d_model, k)
             for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, current_l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(
                out, res, dropout * float(current_l) / total_layers)
            current_l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(
            out, res, dropout * float(current_l) / total_layers)
        current_l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(
            out, res, dropout * float(current_l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training is True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(
                    inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand(
            [-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class QANetCore(nn.Module):
    # add fusion
    def __init__(self, config, PAD=0, dropout=0.1):
        super().__init__()
        self.config = config
        self.encode_fusion = True
        self.match_fusion = True
        self.model_fusion = False
        d_model = config.d_model
        num_head = config.num_head
        self.PAD = PAD
        self.dropout = dropout
        self.num_head = num_head

        self.emb_enc = EncoderBlock(
            conv_num=4, d_model=d_model, num_head=num_head,
            k=7, dropout=dropout)

        cq_att_dim = d_model
        if self.encode_fusion:
            cq_att_dim = d_model * 2
        self.cq_att = CQAttention(d_model=cq_att_dim)

        # encode fusion, match fusion
        cq_resizer_dim = d_model * 4
        if self.encode_fusion:
            cq_resizer_dim *= 2
        if self.match_fusion:
            if self.encode_fusion:
                cq_resizer_dim += d_model * 2
            else:
                cq_resizer_dim += d_model

        self.cq_resizer = Initialized_Conv1d(cq_resizer_dim, d_model)
        self.model_enc_blks = nn.ModuleList(
            [EncoderBlock(conv_num=2, d_model=d_model,
                          num_head=num_head, k=5, dropout=dropout)
             for _ in range(7)])
        # model fusion
        if self.model_fusion:
            self.lstm1 = nn.LSTM(
                input_size=d_model + cq_resizer_dim,
                hidden_size=d_model // 2,
                bidirectional=True, dropout=0.3, batch_first=True)
            self.lstm2 = nn.LSTM(
                input_size=d_model + cq_resizer_dim,
                hidden_size=d_model // 2,
                bidirectional=True, dropout=0.3, batch_first=True)
            self.lstm3 = nn.LSTM(
                input_size=d_model + cq_resizer_dim,
                hidden_size=d_model // 2,
                bidirectional=True, dropout=0.3, batch_first=True)

    def forward(self, C, Q, maskC, maskQ):
        # intput shape is: batch_size * hidden_dim * seq_len
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        if self.encode_fusion:
            Ce = torch.cat([Ce, C], dim=1)
            Qe = torch.cat([Qe, Q], dim=1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        if self.match_fusion:
            X = torch.cat([X, Ce], dim=1)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0
        # ##model fusion
        if self.model_fusion:
            tmp1 = torch.cat([M1, X], dim=1)
            M1, _ = self.lstm1(tmp1.transpose(1, 2))
            M1 = M1.transpose(1, 2)
            tmp2 = torch.cat([M2, X], dim=1)
            M2, _ = self.lstm1(tmp2.transpose(1, 2))
            M2 = M2.transpose(1, 2)
            tmp3 = torch.cat([M3, X], dim=1)
            M3, _ = self.lstm1(tmp3.transpose(1, 2))
            M3 = M3.transpose(1, 2)
        return Ce, Qe, X, M1, M2, M3


class CluePredictor_qanet(nn.Module):
    def __init__(self, config, total_emb_dim, PAD=0, dropout=0.1):
        super().__init__()
        self.config = config
        self.PAD = PAD
        self.dropout = dropout
        # transform embedding
        self.conv1d = Initialized_Conv1d(
            total_emb_dim, config.d_model, bias=False)
        self.high = Highway(2, config.d_model)
        # encoding
        self.qanet_core = QANetCore(config, PAD=PAD, dropout=dropout)
        # is clue word classification
        self.conv = Initialized_Conv1d(
            3 * config.d_model, 1, relu=True, bias=True)
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
        Se, Ae, X, M1, M2, M3 = self.qanet_core(Semb, Aemb, maskS, maskA)
        y_clue_logits = self.linear(self.conv(
            torch.cat((M1, M2, M3), 1))).squeeze()
        y_clue_logits = y_clue_logits * maskS + (1 - maskS) * (-1e30)

        return y_clue_logits, maskA
