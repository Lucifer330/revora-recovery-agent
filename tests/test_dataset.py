from revora.dataset import generate_dataset
from revora.schemas import Split


def test_dataset_reproducible_with_same_seed():
    a = generate_dataset(size=50, seed=42)
    b = generate_dataset(size=50, seed=42)
    assert [p.payment_id for p in a.payments] == [p.payment_id for p in b.payments]
    assert [p.amount_paise for p in a.payments] == [p.amount_paise for p in b.payments]
    assert [p.failure_code for p in a.payments] == [p.failure_code for p in b.payments]
    assert [p.split for p in a.payments] == [p.split for p in b.payments]


def test_dataset_changes_with_different_seed():
    a = generate_dataset(size=50, seed=1)
    b = generate_dataset(size=50, seed=2)
    assert [p.failure_code for p in a.payments] != [p.failure_code for p in b.payments]


def test_splits_are_disjoint_and_cover():
    ds = generate_dataset(size=200, seed=42)
    ids = {p.payment_id for p in ds.payments}
    train = {p.payment_id for p in ds.by_split(Split.TRAIN)}
    dev = {p.payment_id for p in ds.by_split(Split.DEV)}
    held = {p.payment_id for p in ds.by_split(Split.HELD_OUT)}
    assert train.isdisjoint(dev)
    assert train.isdisjoint(held)
    assert dev.isdisjoint(held)
    assert train | dev | held == ids
    assert len(held) > 0 and len(train) > 0
