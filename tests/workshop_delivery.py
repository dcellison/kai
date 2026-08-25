"""Shared delivery-policy fixtures for Workshop contract tests."""

from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy

TELEGRAM_DELIVERY_POLICY = WorkshopDeliveryBindingPolicy(frozenset({"telegram"}))
DISABLED_DELIVERY_POLICY = WorkshopDeliveryBindingPolicy.disabled()
