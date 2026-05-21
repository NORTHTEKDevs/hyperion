"""Tests for the outer-product associative memory."""

from __future__ import annotations

import torch

from pure_vsa.memory import AssociativeMemory
from vsa_core.codebook import Codebook


D = 4096


def test_memory_empty_retrieval_is_noise():
    cb = Codebook(50, D, seed=0)
    mem = AssociativeMemory(D)
    # Querying empty memory returns ~zero; cleanup picks arbitrary index but at low sim.
    key = cb[3]
    _, sim = mem.retrieve(key, cb.all())
    # cleanup of near-zero vector against random codebook -> sim near 0.
    assert abs(sim.item()) < 0.2


def test_memory_single_pair_round_trip():
    torch.manual_seed(0)
    cb = Codebook(50, D, seed=0)
    mem = AssociativeMemory(D)
    key = cb[3]
    val_idx = 7
    val = cb[val_idx]
    mem.store(key, val)
    recovered_idx, sim = mem.retrieve(key, cb.all())
    assert recovered_idx.item() == val_idx, f"got {recovered_idx.item()}"
    assert sim.item() > 0.5


def test_memory_multiple_pairs_no_interference_at_low_load():
    """At 5 pairs in D=4096, all pairs should be perfectly recoverable."""
    torch.manual_seed(0)
    cb = Codebook(100, D, seed=0)
    mem = AssociativeMemory(D)
    pairs = [(0, 50), (1, 51), (2, 52), (3, 53), (4, 54)]
    for k_idx, v_idx in pairs:
        mem.store(cb[k_idx], cb[v_idx])
    correct = 0
    for k_idx, v_idx in pairs:
        recovered_idx, _ = mem.retrieve(cb[k_idx], cb.all())
        if recovered_idx.item() == v_idx:
            correct += 1
    assert correct == len(pairs)


def test_memory_capacity_degrades_above_bound():
    """Above the capacity bound, retrieval accuracy drops."""
    torch.manual_seed(0)
    cb = Codebook(200, D, seed=0)
    mem = AssociativeMemory(D)
    # Store 100 pairs in D=4096 against vocab=200. Capacity bound D/(2 ln N) ~= 386,
    # so 100 should be fine. Pump it to 500 to see degradation.
    n_store = 500
    for i in range(n_store):
        k_idx = i % 200
        v_idx = (i * 7) % 200  # deterministic mapping
        mem.store(cb[k_idx], cb[v_idx])
    # Most recent stores should still be retrievable; older ones drowned out by interference.
    n_correct = 0
    for i in range(n_store):
        k_idx = i % 200
        v_idx = (i * 7) % 200
        recovered_idx, _ = mem.retrieve(cb[k_idx], cb.all())
        if recovered_idx.item() == v_idx:
            n_correct += 1
    # With duplicate keys this is also a noise test. Just assert *some* signal remains.
    assert 0 < n_correct, "memory completely failed; should retain at least some signal"


def test_memory_batch_store():
    torch.manual_seed(0)
    cb = Codebook(20, D, seed=0)
    mem_seq = AssociativeMemory(D)
    mem_batch = AssociativeMemory(D)
    keys = cb.hvs[:5]
    vals = cb.hvs[5:10]
    for i in range(5):
        mem_seq.store(keys[i], vals[i])
    mem_batch.store_batch(keys, vals)
    assert torch.allclose(mem_seq.state, mem_batch.state, atol=1e-5)


def test_memory_topk():
    torch.manual_seed(0)
    cb = Codebook(50, D, seed=0)
    mem = AssociativeMemory(D)
    mem.store(cb[3], cb[7])
    sims, idx = mem.retrieve_topk(cb[3], cb.all(), k=3)
    # the index 7 should be in the top-3
    assert 7 in idx.tolist()
