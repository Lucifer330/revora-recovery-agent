from revora.knowledge import candidate_actions_for, default_knowledge, lookup_policies
from revora.schemas import FailureCategory, PaymentMethodType, RecoveryAction


def test_knowledge_loads_and_is_deterministic():
    a = default_knowledge()
    b = default_knowledge()
    assert a is b or [e.id for e in a] == [e.id for e in b]
    assert len(a) == 6


def test_lookup_by_category():
    entries = lookup_policies(FailureCategory.EXPIRED_PAYMENT_METHOD, PaymentMethodType.CARD)
    assert len(entries) == 1
    actions = {c.action for c in candidate_actions_for(FailureCategory.EXPIRED_PAYMENT_METHOD, PaymentMethodType.CARD)}
    assert RecoveryAction.PAYMENT_METHOD_UPDATE in actions
    assert RecoveryAction.RETRY_LATER not in actions
