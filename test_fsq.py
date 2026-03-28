"""Quick sanity checks for the FSQ module."""

import torch
from fsq import FSQ, ConditionalPrior


def test_basic():
    levels = [5, 5, 5]
    fsq = FSQ(levels)
    print(f"Codebook size: {fsq.codebook_size}")  # 125

    # Simulate spatial: batch=4 images, 49 positions (7×7), D=3 dims
    z = torch.randn(4 * 49, len(levels))
    q, idx = fsq(z)

    print(f"Input shape:      {z.shape}")
    print(f"Quantized shape:  {q.shape}")
    print(f"Indices shape:    {idx.shape}")
    print(f"Index range:      [{idx.min()}, {idx.max()}]")

    loss = q.sum()
    loss.backward()
    assert z.grad is None
    print("Gradient check passed.")


def test_roundtrip():
    levels = [5, 5, 5]
    fsq = FSQ(levels)

    z = torch.randn(16, len(levels))
    q, idx = fsq(z)

    # Decode indices back to codes
    q_decoded = fsq.indices_to_codes(idx)
    assert torch.allclose(q, q_decoded), "Roundtrip encode/decode failed!"
    print("Index roundtrip check passed.")


def test_straight_through():
    levels = [4, 4, 4]
    fsq = FSQ(levels)

    z = torch.randn(8, 3, requires_grad=True)
    q, _ = fsq(z)
    loss = q.sum()
    loss.backward()

    assert z.grad is not None, "No gradient flowed back!"
    print(f"Straight-through gradient norm: {z.grad.norm().item():.4f}")
    print("Straight-through gradient check passed.")


def test_conditional_prior():
    levels = [5, 5, 5]
    fsq = FSQ(levels)
    prior = ConditionalPrior(codebook_size=fsq.codebook_size, num_classes=10)

    labels = torch.randint(0, 10, (16,))
    target_indices = torch.randint(0, fsq.codebook_size, (16,))

    # Loss should be a scalar
    loss = prior.loss(labels, target_indices)
    assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
    loss.backward()
    print(f"Prior CE loss: {loss.item():.4f}")

    # Sampling returns indices in valid range
    samples = prior.sample(labels)
    assert samples.shape == (16,)
    assert samples.min() >= 0 and samples.max() < fsq.codebook_size
    print(f"Sampled indices range: [{samples.min()}, {samples.max()}]")
    print("Conditional prior check passed.")


if __name__ == "__main__":
    test_basic()
    test_roundtrip()
    test_straight_through()
    test_conditional_prior()
    print("\nAll tests passed!")
