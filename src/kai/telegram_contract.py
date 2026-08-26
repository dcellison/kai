"""SDK-free capability declaration for the optional Telegram adapter."""

from kai.workshop.delivery_policy import DeliveryAdapterCapabilities

TELEGRAM_DELIVERY_CAPABILITIES = DeliveryAdapterCapabilities(
    transport="telegram",
    final_text=True,
    preview_streaming=True,
    message_editing=True,
    replies=True,
    threads=True,
    attachments=True,
)
