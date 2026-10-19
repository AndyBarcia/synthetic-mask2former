"""Per-prefix map policy: causality, tree equivalence and differentiability."""
import torch
from mask2former.modeling.object_decoder import ObjectDecoder


def test_map_policy():
    torch.manual_seed(123)
    model = ObjectDecoder(8, 8, 4, 2, 2, 16, map_decoder=True).train()
    embeddings = torch.randn(1, 4, 8, requires_grad=True)
    features = [torch.randn(1, 16, 8, requires_grad=True) for _ in range(3)]
    masks = torch.randn(1, 4, 4, 4)
    regions = model.prepare_regions(masks, [(4, 4)] * 3)
    exclusions = model.prepare_vocabulary_exclusions(masks)
    paths = [[[0, 1], [0, 2, 3]]]
    prev, pos, anc, weights = model.pack_prefix_trees(paths, 4, embeddings.device)
    tree = model(embeddings, features, regions, prev, exclusions, positions=pos, ancestors=anc)
    for path in paths[0]:
        tokens = torch.tensor([[-1] + path])
        linear = model(embeddings, features, regions, tokens, exclusions)
        node = 0
        for step, token in enumerate(tokens[0]):
            if step:
                node = ((prev[0] == token) & anc[0, :, node] & (pos[0] == step)).nonzero().item()
            torch.testing.assert_close(tree[0, node], linear[0, step], atol=3e-6, rtol=3e-6)
    loss = -(tree.log_softmax(-1) * weights).sum() / weights.sum()
    loss.backward()
    assert torch.isfinite(embeddings.grad).all() and embeddings.grad.abs().sum() > 0
    assert features[0].grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name.startswith('inspection_'):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
    # Changing future tokens cannot change earlier next-token predictions.
    a = model(embeddings, features, regions, torch.tensor([[-1, 0, 1]]), exclusions)
    b = model(embeddings, features, regions, torch.tensor([[-1, 0, 3]]), exclusions)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    last = model(embeddings, features, regions, torch.tensor([[-1, 0, 1]]), exclusions, map_last_only=True)
    torch.testing.assert_close(a[:, -1], last[:, -1])
    # Fully overwritten queries do not enter the map summary.
    full = torch.ones_like(regions[0])
    tokens = torch.tensor([[-1, 0, 1]])
    projected = model.input_mask_projection(embeddings)
    changed = projected.clone(); changed[:, 0] += 100
    x = model.map_states(projected, features[0], full, tokens, None, None)
    y = model.map_states(changed, features[0], full, tokens, None, None)
    torch.testing.assert_close(x[:, -1], y[:, -1])
    with torch.no_grad():
        generated = model.generate(embeddings, features, masks, [(4, 4)] * 3)
    assert len(generated) == 1 and len(generated[0][0]) <= 4
    print('map policy tree/linear, future isolation, occlusion, gradients, generation PASS')


if __name__ == '__main__':
    torch.set_num_threads(4)
    test_map_policy()
