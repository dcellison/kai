"""Shared delivery-policy fixtures for Workshop contract tests."""

from kai.telegram_contract import TELEGRAM_DELIVERY_CAPABILITIES
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy

TELEGRAM_DELIVERY_POLICY = WorkshopDeliveryBindingPolicy(
    frozenset({"telegram"}),
    (TELEGRAM_DELIVERY_CAPABILITIES,),
)
DISABLED_DELIVERY_POLICY = WorkshopDeliveryBindingPolicy.disabled()
