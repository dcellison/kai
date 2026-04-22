"""
Evaluation package for the Kai memory pipeline.

Currently houses Layer 1 only (`retrieval`): precision / recall / MRR
on a probe set scored against the live `format_context` log line.
Future layers (end-to-end A/B, longitudinal friction) will land as
sibling modules so they share probe loaders, output formatting, and
the same package home.
"""
