# Workshop agent security and qualification audit

## Scope

This audit covers canonical Workshop agent definitions, per-principal
enablement, direct and shared-channel execution, runtime sponsorship,
delegation, client and internal APIs, memory, workspaces, services, restart,
replay, and operator diagnostics. It treats Telegram and Workshop as clients;
neither client supplies execution authority.

The audit does not claim hostile local multi-user isolation for trusted-host
backend processes. That remains the responsibility of a future isolated worker
boundary. It also does not claim installed live-account coverage for backends
that are not configured on the qualification host.

## Authority boundaries

- An agent definition is immutable, versioned behavior metadata. Its rendered
  instructions explicitly confer no tools, credentials, identity, data access,
  workspace access, or policy authority.
- A human may use an active definition only through a canonical
  `principal_agent_enablement`. The enablement binds exactly one human, agent,
  direct channel, and protected runtime profile.
- A shared-channel attachment records its human sponsor and sponsored runtime
  profile. It is valid only while the sponsor owns the channel and retains a
  matching enabled direct-agent lane.
- Runtime profiles, backend credentials, service names, workspace grants, and
  memory namespaces are resolved by the server from canonical context. Agent
  text cannot broaden them.
- Delegation carries exact caller and target definition revisions, sponsor and
  runtime bindings, a bounded shared-context summary, depth, parentage, and an
  idempotency receipt. Cycles and unbounded depth are rejected before child
  state is created.
- Workshop client sessions bind a human principal. Internal API credentials
  bind a complete canonical execution context and reject caller-supplied
  identity selectors.
- Canonical event, run, message, memory, and delegation records survive restart
  and projection replay. Definition revisions and archive provenance are not
  rewritten when later revisions are activated or definitions are archived.

## Automated evidence

The following suites are the focused security and recovery evidence for this
boundary:

- `tests/test_workshop_agent_definitions.py`: strict definition payloads and
  handles, immutable revisions, exact run provenance, archive behavior,
  replay, and the explicit no-authority instruction contract.
- `tests/test_workshop_agent_enablement.py`: two-human direct-lane isolation,
  cross-principal runtime denial, enable/disable/rebind continuity, and scoped
  event delivery.
- `tests/test_workshop_agent_delegation.py`: explicit delegation, idempotency,
  bounded shared context, cycle rejection, cancellation, terminal settlement,
  restart recovery, and replay.
- `tests/test_workshop_wake_policy.py` and
  `tests/test_workshop_conversation_commands.py`: exact mention routing,
  multi-agent acceptance, detached/unavailable agents, and mutation-free
  rejection.
- `tests/test_workshop_internal_api_contexts.py` and
  `tests/test_internal_api_auth.py`: complete server-bound contexts,
  fail-closed ambiguity, unique credentials, named service scopes, and denial
  of personal memory to shared-channel credentials.
- `tests/test_workshop_settings_workspaces.py` and
  `tests/test_workshop_runtime_pool.py`: protected runtime policy, targeted
  backend changes, runtime-busy rejection, workspace grants, OS-user
  boundaries, named service scopes, and fail-closed unavailable resources.
- `tests/test_workshop_memory_authority.py` and
  `tests/test_workshop_memory_queries.py`: disjoint principal/profile memory
  namespaces, exact provenance, scope enforcement, and fail-closed ambiguous
  legacy state.
- `tests/test_workshop_client_api.py`: authentication before parsing,
  principal/channel non-enumerability, strict mutation schemas, cross-principal
  run/trace/thread/artifact/settings denial, session revocation, and canonical
  SSE replay.

The full test suite, lint/format checks, type checks, dependency audit, and the
Telegram-free core gate remain required CI evidence. The installed
qualification must additionally exercise the lifecycle with two human
principals, at least two agent definitions, and each backend that is actually
available on the host.

## Operator diagnostic

`make install-status` reports one aggregate `Workshop agent authority` line.
It includes definition lifecycle and revision counts, enablements, direct
channels, active and detached shared-channel attachments, runtime sponsorship,
delegation trees, and nonterminal delegation work.

The line becomes `INCOMPLETE` for any of these integrity failures:

- an agent without a definition, or a definition bound across workshops;
- a missing, invalid active, or non-sequential revision history;
- an orphaned/non-agent principal or invalid handle;
- an invalid enabled direct lane or cross-principal runtime binding;
- one runtime namespace owned by more than one human principal;
- a dangling or unsponsored active shared-channel attachment; or
- a delegation whose run parentage, agents, revisions, sponsors, runtimes, or
  tree depth no longer match its durable receipt.

Normal draft or archived definitions, detached historical attachments, and
currently requested/executing delegations are reported but are not themselves
integrity failures.

## Installed qualification limits

Installed completion is evidence for configured accounts, not a claim that
every supported provider account exists. The qualification records which
backends were exercised and why any supported backend was unavailable. Backend
harness tests still cover adapters that cannot be exercised with a live account
on the host.

The trusted-host compatibility runtime warning remains material: a backend
executable writable by its OS user is not a hostile multi-user security
boundary. Canonical principal isolation prevents one Workshop user from
selecting another user's runtime, memory, workspace, or credential context, but
strong containment of a malicious local process requires isolated workers.
