from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20


def test_generate_graph_honors_explicit_false(monkeypatch):
    model = EquiformerV2_OC20.__new__(EquiformerV2_OC20)
    torch.nn.Module.__init__(model)
    model.cutoff = 5.0
    model.max_neighbors = 1000
    model.use_pbc = False
    model.otf_graph = True
    batch = Batch.from_data_list([Data(
        pos=torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        natoms=torch.tensor([2]),
    )])
    def unexpected(*args, **kwargs):
        raise AssertionError("A stored backbone graph should be reused")
    monkeypatch.setattr("nets.equiformer_v2.equiformer_v2_oc20.radius_graph", unexpected)
    edge_index, *_ = model.generate_graph(batch, otf_graph=False)
    assert torch.equal(edge_index, batch.edge_index)


def test_checkpoint_inference_neighbor_default_and_override(monkeypatch):
    from hip import inference_utils

    monkeypatch.setattr(inference_utils.torch, "load", lambda *a, **kw: {
        "hyper_parameters": {"model_config": {"name": "EquiformerV2"}}
    })
    class Model:
        max_neighbors = 32
        def eval(self):
            return self
    monkeypatch.setattr(inference_utils.PotentialModule, "load_from_checkpoint",
                        lambda *a, **kw: SimpleNamespace(potential=Model()))
    assert inference_utils.get_model_from_checkpoint("unused", "cpu").max_neighbors == 1000
    assert inference_utils.get_model_from_checkpoint("unused", "cpu", 17).max_neighbors == 17
    assert inference_utils.get_model_from_checkpoint("unused", "cpu", None).max_neighbors == 32
