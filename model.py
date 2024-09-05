import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, plm):
        super(Model, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = False
        self.use_sim = args.DATA.use_sim
        self.use_ex = args.DATA.use_ex

        linear_in_cnt = 4

        self.MMIP_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        self.in_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
            
        embed_dim = args.MODEL.embed_dim * 2
        if self.use_pos:
            embed_dim += args.MODEL.embed_dim 
        if self.use_ex:
            embed_dim += args.MODEL.embed_dim 
        if self.use_sim:
            embed_dim += 1
        self.fc = nn.Linear(in_features=embed_dim, out_features=args.MODEL.num_classes)

        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.MMIP_linear)
        self._init_weights(self.in_linear)
        if self.use_ex:
            self.ex_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
            self._init_weights(self.ex_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs):
        # get embeddings from the pretrained language model
        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)
        

        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)
            embed_r = torch.div(first_r + last_r, 2)
        else:
            embed_l = last_l
            embed_r = last_r

        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim]

        tar_mask_ls = (segs_ls == 1).long()
        tar_mask_rs = (segs_rs == 1).long()
        

        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)
        H_l = torch.mul(tar_mask_rs.unsqueeze(2), embed_r)
       

        h_s = torch.mean(embed_l, dim=1)  # sentence meaning
        h_t = torch.mean(H_t, dim=1)  # target word context meaning
        h_l = torch.mean(H_l, dim=1)  # literal meaning
        
        
        pdist = torch.nn.PairwiseDistance(p=1)
        sim = pdist(h_t, h_l)

        h_Mmip = torch.cat((h_t, h_l, torch.abs(h_t - h_l), torch.mul(h_t, h_l)), dim=-1)
        h_in = torch.cat((h_s, h_t, torch.abs(h_s - h_t), torch.mul(h_s, h_t)), dim=-1)
        h_ex = torch.cat((h_s, h_l, torch.abs(h_s - h_l), torch.mul(h_s, h_l)), dim=-1)

        h_Mmip = self.MMIP_linear(h_Mmip)
        if self.use_sim:
            sim = sim.unsqueeze(1)
            h_Mmip = torch.cat((h_Mmip, sim), dim=-1)
        h_in = self.in_linear(h_in)
        if self.use_ex:
            h_ex = self.ex_linear(h_ex)
            h_IEspv = torch.cat((h_in, h_ex), dim=-1)
        else:
            h_IEspv = h_in
        final = torch.cat((h_Mmip, h_IEspv), dim=-1)

        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((final, h_p), dim=-1)

        final = self.dropout3(final)
        out = self.fc(final)  # [batch_size, num_classes]
        return out


class Model_base(nn.Module):
    def __init__(self, args, plm, vua_all=False):
        super(Model_base, self).__init__()
        self.plm_config = plm.config
        self.num_classes = args.MODEL.num_classes
        self.plm = plm
        for param in self.plm.parameters():
            param.requires_grad = True
        self.dropout1 = nn.Dropout(args.MODEL.dropout)
        self.dropout2 = nn.Dropout(args.MODEL.dropout)
        self.first_last_avg = args.MODEL.first_last_avg
        self.use_context = args.DATA.use_context
        self.use_pos = vua_all

        linear_in_cnt = 4

        self.MIP_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        self.SPV_linear = nn.Linear(in_features=args.MODEL.embed_dim * linear_in_cnt, out_features=args.MODEL.embed_dim)
        embed_dim = args.MODEL.embed_dim * 2
        if self.use_pos:
            embed_dim += args.MODEL.embed_dim 
        self.fc = nn.Linear(in_features=embed_dim, out_features=args.MODEL.num_classes)
    
        self.dropout3 = nn.Dropout(args.MODEL.dropout)
        self._init_weights(self.fc)
        self._init_weights(self.MIP_linear)
        self._init_weights(self.SPV_linear)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.plm_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, ids_ls, segs_ls, att_mask_ls, ids_rs, att_mask_rs, segs_rs):
        # get embeddings from the pretrained language model
        if self.use_context:
            out_l = self.plm(ids_ls, token_type_ids=segs_ls,
                             attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, token_type_ids=segs_rs,
                             attention_mask=att_mask_rs, output_hidden_states=True)
        else:
            out_l = self.plm(ids_ls, attention_mask=att_mask_ls, output_hidden_states=True)
            out_r = self.plm(ids_rs, attention_mask=att_mask_rs, output_hidden_states=True)

        last_l = out_l.hidden_states[-1]  # last layer hidden states of the PLM
        first_l = out_l.hidden_states[1]  # first layer hidden states of the PLM

        last_r = out_r.hidden_states[-1]
        first_r = out_r.hidden_states[1]

        if self.first_last_avg:
            embed_l = torch.div(first_l + last_l, 2)
            embed_r = torch.div(first_r + last_r, 2)
        else:
            embed_l = last_l
            embed_r = last_r

        embed_l = self.dropout1(embed_l)  # [batch_size, max_left_len, embed_dim]
        embed_r = self.dropout2(embed_r)  # [batch_size, max_left_len, embed_dim]

        # H_l ==> H_t for target;  H_r ==> H_b for basic meaning
        tar_mask_ls = (segs_ls == 1).long()
        tar_mask_rs = (segs_rs == 1).long()
        H_t = torch.mul(tar_mask_ls.unsqueeze(2), embed_l)
        H_b = torch.mul(tar_mask_rs.unsqueeze(2), embed_r)

        h_c = torch.mean(embed_l, dim=1)  # context representation
        h_t = torch.mean(H_t, dim=1)  # contextualized target meaning
        h_b = torch.mean(H_b, dim=1)  # basic meaning
 
        h_mip = torch.cat((h_t, h_b, torch.abs(h_t - h_b), torch.mul(h_t, h_b)), dim=-1)
        h_spv = torch.cat((h_c, h_t, torch.abs(h_c - h_t), torch.mul(h_c, h_t)), dim=-1)

        h_mip = self.MIP_linear(h_mip)
        h_spv = self.SPV_linear(h_spv)

        final = torch.cat((h_mip, h_spv), dim=-1)
        if self.use_pos:
            pos_mask = (segs_rs == 3).long()
            H_p = torch.mul(pos_mask.unsqueeze(2), embed_r)
            h_p = torch.mean(H_p, dim=1)
            final = torch.cat((final, h_p), dim=-1)

        final = self.dropout3(final)
        out = self.fc(final)  # [batch_size, num_classes]
        return out

if __name__ == "__main__":
    pass
