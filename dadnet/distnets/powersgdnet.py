import torch
from dadnet.distnets.distnet import DistNet


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


class PowerSgdNet(DistNet):
    def __init__(self, *networks, rank=2, **kwargs):
        self.rank = rank
        self.Qs = dict()
        self.Ps = dict()
        super(PowerSgdNet, self).__init__(*networks, **kwargs)

    def recompute_gradients(self):
        for site, network in enumerate(self.networks):
            for m_i, module in enumerate(self.reverse_modules(network)):
                if not hasattr(module, "_order"):
                    continue
                mname = module._order
                if mname not in self.Ps.keys():
                    self.Qs[mname] = dict()
                    self.Ps[mname] = dict()
                for p_i, parameter in enumerate(module.parameters()):
                    grad = parameter.grad
                    if p_i not in self.Qs[mname].keys():
                        self.Qs[mname][p_i] = dict()
                        self.Ps[mname][p_i] = dict()
                    if site not in self.Qs[mname][p_i].keys():
                        Q = torch.randn(parameter.shape[-1], self.rank).to(
                            parameter.device
                        )
                    else:
                        Q = self.Qs[mname][p_i][site]
                    # if len(grad.shape) == 1:
                    #    continue
                    P = grad @ Q
                    P = gram_schmidt(P)
                    Q = grad.t() @ P
                    self.Qs[mname][p_i][site] = Q.clone()
                    self.Ps[mname][p_i][site] = P.clone()
        seed_network = self.networks[0]
        for m_i, seed_module in enumerate(self.reverse_modules(seed_network)):
            if not hasattr(seed_module, "_order"):
                continue
            seed_mname = seed_module._order
            for p_i, _ in enumerate(seed_module.parameters()):
                Q_agg = []
                P_agg = []

                for site, network in enumerate(self.networks):
                    Q_agg.append(self.Qs[seed_mname][p_i][site])
                    P_agg.append(self.Ps[seed_mname][p_i][site])

                Q_agg = torch.stack(Q_agg, -1).mean(-1)
                P_agg = torch.stack(P_agg, -1).mean(-1)
                agg_grad = P_agg @ Q_agg.t()
                for site, network in enumerate(self.networks):
                    module = list(self.reverse_modules(network))[m_i]
                    parameter = list(module.parameters())[p_i]
                    try:
                        parameter.grad = agg_grad.clone()
                    except KeyError:
                        continue
