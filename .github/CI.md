# Continuous integration validation

Kai's pull-request validation preserves the complete pytest suite while running its slowest domains concurrently.

## Change scopes

`scripts/ci_change_scope.py` selects validation conservatively:

- documentation-only changes use the no-runtime fast path;
- Workshop client or generated-asset-only changes run the Workshop client checks;
- every other change runs client checks, Python quality checks, and every pytest shard;
- pushes to `main` force every validation lane, providing a complete post-merge backstop;
- dependency inputs additionally run the independent dependency audit.

Unknown or empty change input fails closed to complete validation.

## Complete pytest gate

`scripts/ci_test_shard.py` assigns every `tests/test_*.py` module to exactly one stable shard:

- `core`: general application, adapter, configuration, installation, and integration tests;
- `memory`: `test_memory*` and `test_eval_*` modules;
- `workshop`: `test_workshop_*` modules.

The shards run concurrently with fail-fast disabled, so failures remain independently visible. Unit coverage proves that their sets are exhaustive and disjoint. The required `check` job succeeds only after the quality job and every selected shard succeed.

The separate `core-workshop` job remains intentional: it installs Kai without Telegram and proves the transport-independent architecture in that distinct dependency environment.

Run a shard locally with:

```bash
python scripts/ci_test_shard.py core
python scripts/ci_test_shard.py memory
python scripts/ci_test_shard.py workshop
```
