"""Operator-facing smoke commands for verifying a Kai install.

Each smoke command runs a small, idempotent verification of one
runtime surface; see individual modules for the contract. The
package layout matches `kai.eval.*`: each smoke is a directly-
invocable module (`python -m kai.smoke.<name>`), not a subcommand
dispatched through `kai.smoke.__main__`.
"""
