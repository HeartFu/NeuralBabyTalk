import torch
import torch.nn as nn
from torch.autograd import Variable

from misc.relation.fc import FCNet
from misc.relation.graph_att import GAttNet


class ImplicitRelationEncoder(nn.Module):
    def __init__(self, v_dim, out_dim, dir_num, pos_emb_dim,
                 nongt_dim, num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True):
        # q_dim is question embedding, need remove
        super(ImplicitRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.out_dim = out_dim
        self.residual_connection = residual_connection
        self.num_steps = num_steps
        print("In ImplicitRelationEncoder, num of graph propogate steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])
        else:
            self.v_transform = None
        # in_dim = out_dim+q_dim
        in_dim = out_dim
        self.implicit_relation = GAttNet(dir_num, 1, in_dim, out_dim,
                                     nongt_dim=nongt_dim,
                                     label_bias=label_bias,
                                     num_heads=num_heads,
                                     pos_emb_dim=pos_emb_dim)

    def forward(self, v, position_embedding):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            position_embedding: [batch_size, num_rois, nongt_dim, emb_dim]

        Returns:
            output: [batch_size, num_rois, out_dim,3]
        """
        # [batch_size, num_rois, num_rois, 1]
        imp_adj_mat = Variable(
            torch.ones(
                v.size(0), v.size(1), v.size(1), 1)).to(v.device)
        imp_v = self.v_transform(v) if self.v_transform else v

        for i in range(self.num_steps):
            # v_cat_q = q_expand_v_cat(q, imp_v, mask=True)
            v_cat_q = imp_v
            imp_v_rel = self.implicit_relation.forward(v_cat_q,
                                                       imp_adj_mat,
                                                       position_embedding)
            if self.residual_connection:
                imp_v += imp_v_rel
            else:
                imp_v = imp_v_rel
        return imp_v

class ExplicitRelationEncoder(nn.Module):
    def __init__(self, v_dim, out_dim, dir_num, label_num,
                 nongt_dim=20, num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True):
        super(ExplicitRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.out_dim = out_dim
        self.num_steps = num_steps
        self.residual_connection = residual_connection
        print("In ExplicitRelationEncoder, num of graph propogation steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])
        else:
            self.v_transform = None
        in_dim = out_dim
        self.explicit_relation = GAttNet(dir_num, label_num, in_dim, out_dim,
                                     nongt_dim=nongt_dim,
                                     num_heads=num_heads,
                                     label_bias=label_bias,
                                     pos_emb_dim=-1)

    def forward(self, v, exp_adj_matrix):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            q: [batch_size, q_dim]
            exp_adj_matrix: [batch_size, num_rois, num_rois, num_labels]

        Returns:
            output: [batch_size, num_rois, out_dim]
        """
        exp_v = self.v_transform(v) if self.v_transform else v

        for i in range(self.num_steps):
            # v_cat_q = q_expand_v_cat(q, exp_v, mask=True)
            v_cat_q = exp_v
            exp_v_rel = self.explicit_relation.forward(v_cat_q, exp_adj_matrix)
            if self.residual_connection:
                exp_v += exp_v_rel
            else:
                exp_v = exp_v_rel
        return exp_v
