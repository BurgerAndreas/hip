"""Add an optional smooth polynomial envelope to edge messages (backbone + Hessian head)."""
from pathlib import Path
import sys
root = Path(sys.argv[1])

def sub(path, old, new, count=1):
    p = root / path; s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))

ENV_FN = '''

def poly_envelope(x, p=5):
    """Smooth cutoff u(x), x = d / cutoff: u(0)=1, u(1)=u'(1)=u''(1)=0, zero for x >= 1 (DimeNet)."""
    x = x.clamp(min=0.0, max=1.0)
    u = 1 - (p + 1) * (p + 2) / 2 * x**p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
    return u * (x < 1).to(u.dtype)
'''
oc = 'nets/equiformer_v2/equiformer_v2_oc20.py'
s = (root / oc).read_text()
first_def = s.index('\nclass ')
(root / oc).write_text(s[:first_def] + ENV_FN + s[first_def:])
sub(oc, '        fully_connected_hessian=True,\n        device=None,\n        **kwargs,\n    ):\n        super().__init__()',
         '        fully_connected_hessian=True,\n        edge_envelope=False,\n        envelope_exponent=5,\n        device=None,\n        **kwargs,\n    ):\n        super().__init__()\n        self.edge_envelope = bool(edge_envelope)\n        self.envelope_exponent = int(envelope_exponent)')
sub(oc, '            edge_distance = self.distance_expansion(edge_distance)',
         '''            env_bb = poly_envelope(edge_distance / self.cutoff, self.envelope_exponent) if self.edge_envelope else None
            self.edge_degree_embedding._edge_envelope = env_bb
            for _blk in self.blocks:
                _blk.ga._edge_envelope = env_bb
            edge_distance = self.distance_expansion(edge_distance)''')
sub(oc, '''                edge_distance_hessian = self.distance_expansion_hessian(
                    edge_distance_hessian
                )
''', '''                env_h = (
                    poly_envelope(edge_distance_hessian / self.hessian_graph_cutoff, self.envelope_exponent)
                    if (self.edge_envelope and not self.fully_connected_hessian) else None
                )
                for _blk in self.hessian_layers:
                    _blk.ga._edge_envelope = env_h
                edge_distance_hessian = self.distance_expansion_hessian(
                    edge_distance_hessian
                )
''')
sub(oc, '                hessian = self._get_hessian_from_features(',
         '''                if env_h is not None:
                    l012_edge_features_3x3 = l012_edge_features_3x3 * env_h.view(-1, *([1] * (l012_edge_features_3x3.dim() - 1)))
                hessian = self._get_hessian_from_features(''')

tb = 'nets/equiformer_v2/transformer_block.py'
s = (root / tb).read_text()
i = s.index('        if not attn_wo_sigmoid:\n')
j = s.index('\n', s.index('torch_geometric.utils.softmax', i)) + 1
old_block = s[i:j]
assert 'alpha = torch_geometric.utils.softmax(src=alpha, index=edge_index[1])' in old_block, old_block
new_block = old_block.replace('alpha = torch_geometric.utils.softmax(src=alpha, index=edge_index[1])',
    '''_env = getattr(self, "_edge_envelope", None)
            if _env is not None:
                # smooth attention: env * softmax(a + log env) -> continuous normaliser, weight -> 0 at cutoff
                _env = _env.to(alpha.dtype).unsqueeze(-1)
                alpha = torch_geometric.utils.softmax(src=alpha + torch.log(_env.clamp_min(1e-30)), index=edge_index[1]) * _env
            else:
                alpha = torch_geometric.utils.softmax(src=alpha, index=edge_index[1])''')
(root / tb).write_text(s[:i] + new_block + s[j:])

ib = 'nets/equiformer_v2/input_block.py'
sub(ib, '        x_edge_m_0 = self.rad_func(x_edge)\n',
         '        x_edge_m_0 = self.rad_func(x_edge)\n        _env = getattr(self, "_edge_envelope", None)\n        if _env is not None:\n            x_edge_m_0 = x_edge_m_0 * _env.to(x_edge_m_0.dtype).unsqueeze(-1)\n')
print('patched')
