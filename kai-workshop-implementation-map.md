# Kai Workshop: Phase 0 Implementation Map

**Status:** Active migration plan; first canonical conversation delivery authority live
**Date:** 2026-08-12
**Scope:** Map and execute the current Kai implementation's migration onto the proposed Kai Workshop architecture.

## 1. Naming and scope

**Kai Workshop** is the product: the server-hosted collaboration environment used through desktop, web, Telegram, and future clients.

A **workspace** keeps its existing Kai meaning: a project or filesystem execution context, usually a repository or directory in which an agent works. The product name no longer competes with this established term.

For the target domain model, this map uses:

- **workshop**: the top-level collaboration and security boundary inside the product;
- **channel**: an ordered conversation container;
- **project workspace**: the filesystem/repository context selected for agent work;
- **principal**: a human, agent, service, or integration identity;
- **run**: one durable unit of requested agent work;
- **attempt**: one execution of a run through a harness and runtime.

The first installation may contain exactly one workshop, but identifiers and foreign keys should exist from the first schema so a second workshop does not require redefining identity, authorization, or event ordering.

## 2. Executive assessment

Kai can move toward Kai Workshop incrementally without replacing the working Telegram system or the five backend harnesses.

The strongest existing foundations are:

- a long-running server-authoritative service;
- a durable SQLite store for configuration state, jobs, and accepted Telegram updates;
- five functioning backend harness adapters (`claude`, `codex`, `goose`, `opencode`, and `pi`);
- per-user process isolation and an administrator-owned backend registry;
- principal-bound internal API credentials with explicit scopes;
- per-user history, files, preferences, memory, settings, jobs, and project workspace access;
- working Telegram, GitHub, generic webhook, scheduling, memory, file, and service-proxy paths.

The largest missing foundation is not another harness abstraction. It is a canonical collaboration and execution model independent of Telegram. Today, a Telegram `chat_id` is used as a conversation key, runtime pool key, history partition, settings namespace, job owner, file partition, and often an indirect user key. This works for private Telegram chats, where the human's Telegram user ID and chat ID are normally equal, but it cannot represent Workshop channels, multiple human participants, named agents, threads, or the distinction between an author and a conversation.

The first implementation work should therefore establish durable Workshop, principal, channel, message, and event identities beside the current paths. Telegram should become the first adapter onto that model, not be removed or rewritten in one step.

## 3. Current architecture

```text
Telegram updates
    |
    v
webhook.py durable Telegram queue (SQLite)
    |
    v
python-telegram-bot handlers in bot.py
    |                   |                    |
    |                   |                    +--> JSONL history
    |                   +--> SQLite settings/jobs/session metadata
    v
SubprocessPool keyed by Telegram chat_id
    |
    +--> Claude Code backend -----+
    +--> Codex backend ----------- |
    +--> Goose/ACP backend ------- +--> local_process runtime embedded in each adapter
    +--> OpenCode/ACP backend ---- |
    +--> Pi RPC backend ----------+
    |
    +--> project workspace, memory, files, internal API, service proxy

GitHub/generic webhooks and scheduled jobs
    +--> Telegram delivery and/or the same per-chat backend pool
```

The outer process is already a control plane in the practical sense: it authenticates callers, owns configuration, routes work, starts agent processes, brokers selected services, and delivers results. It is not yet a distinct control-plane architecture because transport, domain behavior, persistence, scheduling, and execution orchestration remain in one Python service and share Telegram-shaped identifiers.

## 4. Current sources of truth

| Concern | Current authority | Durable key | Important limitation |
|---|---|---|---|
| Installed harnesses | Protected `/etc/kai/backends.yaml` through [`backend_registry.py`](src/kai/backend_registry.py) | Backend ID | Describes only `local_process`; it is a machine capability registry, not a durable agent registry. |
| Human configuration | Protected `users.yaml` loaded into `UserConfig` in [`config.py`](src/kai/config.py) | Telegram user ID | A Telegram identity is the primary human identity. There is no transport-independent principal ID. |
| Telegram authorization | Sender user ID checked in [`bot.py`](src/kai/bot.py) | Telegram user ID | Authorization correctly uses the sender, but later state routing usually uses the chat ID. |
| Conversation routing | Telegram effective chat in [`bot.py`](src/kai/bot.py) | Telegram chat ID | Author and conversation are separate Telegram concepts but collapse in private-chat operation. |
| Accepted inbound Telegram work | `telegram_update_queue` in [`sessions.py`](src/kai/sessions.py) | Telegram update ID | Strong Telegram-specific durability pattern; not yet a general command inbox. |
| Transcript | Per-user/date JSONL in [`history.py`](src/kai/history.py) | Chat ID + timestamp | No stable message ID, channel ID, thread ID, edit lineage, delivery state, or transactional relation to SQLite. |
| Harness session metadata | `sessions` table plus live backend instance | Chat ID | The stored session ID is used for statistics and clearing; process/session reconstruction still belongs to the live backend. |
| Durable run lifecycle | Workshop event log plus replayed `runs` and `run_attempts` projections | Run ID + attempt ID | Production-unused acceptance and fenced execution authority exist; the cutover review still holds live activation until atomic acceptance, terminal-result finalization, user-visible failure/cancellation outcomes, and replay suppression are integrated. |
| Active harness process | `SubprocessPool` in [`pool.py`](src/kai/pool.py) | Chat ID | One implicit agent process per conversation key; durable agent and run identities now exist but are not connected to process, attempt, lease, or worker ownership. |
| User settings | Generic `settings` rows in [`sessions.py`](src/kai/sessions.py) | Namespaced strings containing chat ID | Flexible but not typed, versioned, or attached to transport-independent principals/channels. |
| Scheduled jobs | `jobs` table and APScheduler registration | Integer job ID + chat ID | Job definitions survive restart, but executions are not durable runs with attempts and event history. |
| Project workspace selection | Settings, `allowed_workspaces`, `workspace_history`, static configuration | Chat ID + filesystem path | A project workspace is not a first-class durable object and is coupled to the current chat. |
| Memory | Qdrant/Mem0 plus per-user files and memory project registry | Stringified chat ID and optional project ID | Useful principal/project scoping exists, but principal IDs are Telegram-derived and no channel/run memory layer exists. |
| Files | Per-chat paths beneath the Kai data directory | Chat ID + generated filename | No immutable artifact identity, provenance record, channel visibility, or delivery lifecycle. |
| Internal agent authority | Process-local credentials in [`internal_api_auth.py`](src/kai/internal_api_auth.py) | Credential -> fixed chat ID/scopes | Good capability foundation, but the principal is still represented by chat ID and credentials do not survive control-plane restart. |
| External services | Static service definitions and per-user grants in [`services.py`](src/kai/services.py) | Service name + chat ID | A useful connector/secret-broker precursor, without attempt-scoped grants or durable access events. |
| GitHub events | Verified webhook handling and per-user routing in [`webhook.py`](src/kai/webhook.py) | Repository/event payload + configured user | Notifications and automation are not projected into canonical channels/events. |
| Outbound Telegram delivery | Direct Bot API calls from handlers, jobs, and webhook paths | Telegram chat ID | There is no general durable outbox or delivery-attempt record. JSONL may be written before delivery succeeds. |

## 5. Identity and conversation findings

### 5.1 Sender identity and conversation identity are already distinct at ingress

Telegram exposes an effective user and an effective chat. Kai's authorization wrapper checks the user ID, while message processing, pool selection, history, jobs, settings, and files use the chat ID. In private chats those numbers usually coincide, hiding the distinction.

Kai Workshop must preserve the distinction explicitly:

```text
Telegram user ID  -> external_identity -> human principal
Telegram chat ID  -> transport_binding -> Workshop channel
Telegram message  -> external_message reference -> canonical message
```

This also preserves the existing security rule that a Telegram group used only as a GitHub notification destination does not become an inbound principal.

### 5.2 The current "agent" is implicit

The selected backend, model, process, identity files, memory, and permissions collectively behave like an agent, but there is no durable named agent object. They are resolved primarily from the owning user's configuration and current chat state.

The first Workshop schema needs a durable default agent definition, even if the UI initially displays only one agent named Kai. Backend and model selection become policy on that agent or its channel attachment, rather than identity itself. Switching from Codex to Pi must not create a different agent identity.

### 5.3 A project workspace is not the collaboration boundary

The current workspace path controls filesystem execution context and workspace-specific model/prompt settings. It should map to a Workshop project workspace or checkout, not to the top-level workshop or channel. A channel may refer to one project workspace, change project workspace, or host discussion with no active project workspace.

## 6. Module-to-target mapping

| Current module or surface | Target concept | Disposition |
|---|---|---|
| [`bot.py`](src/kai/bot.py) | Telegram command adapter and Telegram renderer | **Extract gradually.** Keep existing handlers while moving accepted commands and resulting messages through transport-neutral services. |
| [`webhook.py`](src/kai/webhook.py) | Telegram/GitHub/generic ingress adapters plus current HTTP API | **Split by boundary later.** Authentication and transport parsing remain adapters; domain commands move behind a service layer. |
| [`sessions.py`](src/kai/sessions.py) | Current SQLite persistence | **Extend first.** Add Workshop event and projection tables behind a narrow store interface; do not migrate databases in the first slice. |
| [`history.py`](src/kai/history.py) | Legacy transcript store and memory-provenance source | **Preserve during migration.** Shadow-write canonical messages, verify parity, then make JSONL a compatibility export or retained diagnostic source. |
| [`backend.py`](src/kai/backend.py) | Existing harness-neutral conversational contract and context assembly | **Wrap and evolve.** It is the practical precursor to a Harness Driver, but its event vocabulary is text/session-centric and lacks capabilities, tool events, approvals, usage, and artifacts. |
| `claude.py`, `codex.py`, `goose.py`, `opencode.py`, `pi.py`, and [`acp.py`](src/kai/acp.py) | Five Harness Drivers | **Preserve equally.** Derive a common capability/normalized-event contract from all five implementations; do not choose one as semantically privileged. |
| [`backend_registry.py`](src/kai/backend_registry.py) | Installed harness/runtime capability registry | **Preserve.** Later extend descriptors and versions; do not let user or agent configuration supply executable paths. |
| [`pool.py`](src/kai/pool.py) | Early execution coordinator | **Wrap first, decompose later.** It already resolves identity, configuration, workspace, credentials, lifecycle, and routing. A future orchestrator should use run/attempt IDs rather than chat IDs. |
| Process creation and teardown inside each backend | Server-process Runtime Backend | **Extract after the event foundation.** The current trusted-host runtime is a migration mode; container work should not block the first Workshop conversation slice. |
| [`internal_api_auth.py`](src/kai/internal_api_auth.py) | Scoped execution credentials | **Generalize later.** Replace chat-bound principals with durable principal/run/attempt claims and make issuance auditable. |
| [`cron.py`](src/kai/cron.py) and `jobs` | Schedules and workflow triggers | **Adapt after runs exist.** A firing job should eventually create a durable run rather than directly call a pool and Telegram. |
| [`services.py`](src/kai/services.py) | Connector registry and early secret broker | **Preserve and wrap.** Later authorize against attempt-scoped grants and emit access audit events. |
| Memory modules and `memory_projects` | Core, principal, and project memory | **Preserve.** Introduce durable Workshop principal/project/channel/run references without replacing Qdrant in the first phases. |
| Per-chat file directories | Upload and artifact staging | **Preserve initially.** Add canonical artifact metadata and provenance before changing object storage. |
| `users.yaml`, `workspaces.yaml`, `memory-projects.yaml` | Operator-managed installation policy/bootstrap | **Retain.** Workshop state may become mutable domain data, but protected files remain the safe bootstrap and installation-policy surface during migration. |

## 7. Harness and runtime assessment

### 7.1 Existing Harness Driver precursor

`AgentBackend` already normalizes:

- asynchronous streaming text;
- a terminal success/error response;
- a harness session identifier;
- model, provider, timeout, and project workspace selection;
- restart, shutdown, force-kill, and liveness;
- per-user context, memory, preferences, files, and internal API access.

That contract is sufficient for Telegram conversation but not yet for the Workshop run inspector. It does not expose normalized tool calls, approval requests, usage, artifacts, patches, capability discovery, or distinct lifecycle events.

The next contract must be derived from the behavior of all five harnesses. Capability differences should remain visible. A driver is allowed to report that it cannot resume, branch, expose structured tools, or provide exact usage.

### 7.2 Existing Runtime Backend precursor

Every backend currently owns both harness protocol behavior and local process mechanics:

- command construction;
- `sudo -H -u` isolation;
- environment sanitation;
- subprocess creation;
- stdout/stderr framing;
- descendant discovery and termination;
- temporary-directory and project-workspace setup.

This is the seam for a future `local_process` Runtime Backend, but extracting it now would delay the user-visible Workshop foundation and risk five working integrations. Runtime extraction belongs after canonical runs and normalized run events exist, because those concepts define what the runtime is executing and reporting.

The current server-process mode remains a trusted-host compatibility runtime. Container isolation remains the intended personal-production boundary for Workshop, but it is not a prerequisite for recording and replaying one canonical conversation.

## 8. Durability and ordering findings

### 8.1 Strong reusable pattern: the Telegram update queue

The durable inbound queue persists an update before acknowledging it, deduplicates Telegram retries by `update_id`, atomically claims work, and requeues interrupted processing at startup. This is the best current template for a general command inbox.

It should not simply be renamed into the event log. Commands and events have different meaning:

- the inbox records external intent that may still be rejected;
- the event log records accepted authoritative facts;
- projections provide queryable current state;
- a delivery outbox records effects still owed to external transports.

### 8.2 Transcript writes are durable but not transactional domain events

JSONL history is appended immediately and provides useful recovery and memory provenance. It has no database transaction with Telegram queue completion, session updates, memory extraction, or outbound delivery. An assistant record can be written before the final Telegram delivery attempt succeeds.

The migration must therefore avoid treating existing JSONL ordering as proof of external delivery. Canonical message acceptance and transport delivery need separate states.

### 8.3 Scheduled definitions are durable; executions are not

Job definitions survive restart and are re-registered with APScheduler. A firing job directly invokes Telegram or the backend pool. There is no durable run/attempt record that can distinguish queued, started, waiting, completed, failed, retried, or delivered execution.

Workshop should leave scheduling behavior alone until the run model exists, then make schedule firing create the same run command used by a human message.

## 9. Migration seams

### Seam A: Telegram ingress to canonical commands

After authentication and transport validation, translate an accepted Telegram message into a transport-neutral `messages.create` command carrying:

- authenticated human principal ID;
- bound channel ID;
- external update and message references;
- idempotency key;
- text/media references;
- timestamp and transport metadata.

During shadow operation, existing handlers continue to own behavior. The canonical recorder observes the same accepted input but does not alter routing or responses.

### Seam B: Agent output to canonical messages and run events

Wrap `pool.send()` so its existing `StreamEvent` values can produce canonical run/output events while still feeding the Telegram renderer. Initially, only stable facts should be durable: run accepted, run started, visible text snapshot/final result, failure, cancellation, and session reference. Tool-specific normalization follows only after each driver can supply reliable evidence.

### Seam C: Canonical messages to transport delivery

Separate creation of an assistant message from delivery to Telegram. The eventual model is:

```text
message.created -> delivery.requested -> delivery.succeeded | delivery.failed
```

The first shadow slice may continue direct Telegram delivery, but authority must not cut over until a durable delivery outbox and idempotent retry policy exist.

### Seam D: Chat-keyed processes to agent/channel sessions

Introduce stable agent and channel IDs before changing pool keys. A compatibility mapping can continue to resolve one `(channel_id, agent_id)` pair to the current Telegram `chat_id` pool entry. Later, the pool/orchestrator can key processes by durable session or attempt ID.

### Seam E: JSONL history to canonical projections

Keep writing JSONL while canonical events are shadowed. Add a parity diagnostic that compares canonical message order/content with the JSONL records for the bound Telegram channel. Only after sustained parity should context assembly and transcript views read from canonical projections.

### Seam F: Current files to artifacts

Retain current per-user file storage. Add artifact metadata containing a stable artifact ID, owning workshop/channel/principal, content hash, media type, provenance, and current local path. Moving bytes to object storage is a later storage decision.

## 10. Recommended architectural decisions

These recommendations narrow the original design's open decisions for the first Workshop slice.

1. **Multi-principal schema from day one.** The first deployment can remain personal, but principals and memberships must not be omitted. Kai's existing multi-user and OS-user isolation makes a single-principal schema a regression.
2. **One default workshop at migration.** Create one workshop automatically for an existing installation. Do not expose multi-workshop administration in the first UI.
3. **SQLite first, behind an event-store interface.** Kai already operates a single-server transactional SQLite database in WAL mode. Add the initial event log and projections there. PostgreSQL remains the target when concurrency or deployment topology requires it; adopting it before validating the domain would add operational work without improving the first slice.
4. **One Telegram chat binding to one channel.** Existing private-chat history maps to a default direct channel. Notification-only Telegram groups remain delivery destinations unless explicitly configured as interactive bindings later.
5. **One durable default agent named Kai.** Preserve current behavior while separating agent identity from the selected harness and model. All five harnesses remain equal selectable drivers.
6. **Continue one session per channel/agent by default.** Preserve current conversational continuity. `/new` begins a new harness session without deleting channel history. Explicit branching and cross-channel session policies can follow capability discovery.
7. **Trusted local process during migration.** Keep the currently working runtime while building the domain/event seam. Container execution begins when durable runs and attempts can describe, observe, and recover it.
8. **Telegram remains the first write client.** The first web surface is read-only diagnostics/replay. It should not accept commands until server authorization, idempotency, and delivery behavior are proven independently of Telegram.
9. **No required Redis, object store, or separate queue yet.** SQLite is authoritative for the first personal-server slice. New components require measured need or a later deployment phase.

## 11. First implementation slice after this map

The first code milestone should be **canonical conversation shadowing**. It establishes the model without making it authoritative prematurely.

### 11.1 Minimal domain records

- `workshops`
- `principals`
- `external_identities`
- `workshop_memberships`
- `channels`
- `channel_memberships`
- `channel_bindings`
- `agents`
- `channel_agents`
- `messages`
- `event_log`
- projection checkpoint/version metadata

Identifiers should be opaque, globally unique strings. Telegram numeric identifiers remain external references, never primary domain identifiers.

### 11.2 Minimal event vocabulary

- `workshop.created`
- `workshop.member_added`
- `principal.created`
- `external_identity.bound`
- `channel.created`
- `channel.member_added`
- `transport.channel_bound`
- `agent.created`
- `channel.agent_attached`
- `message.created`
- `artifact.created`
- `delivery.requested`
- `delivery.succeeded`
- `delivery.failed`
- `run.accepted`
- `run.started`
- `run.completed`
- `run.failed`
- `run.cancelled`

The delivery foundation adds only the request and outcome facts the system can define consistently; mutable leases and attempts remain operational outbox state rather than replayed collaboration events. The initial run family likewise records only backend-neutral lifecycle facts. The schema must not invent detailed tool or backend-worker events that the current harnesses cannot yet emit consistently.

### 11.3 Small PR sequence

Every PR that introduces a shadow write, compatibility adapter, fallback, dual-read path, feature flag, or legacy projection must also add or update an entry in the transition retirement ledger in section 17. Transitional work is incomplete until its removal conditions and deletion scope are recorded.

1. **Domain and store contract, unused by production:** typed identifiers, versioned event envelope, SQLite schema/migrations, append/idempotency/projection tests.
2. **Bootstrap migration:** create the default workshop, human principals, default Kai agent, and Telegram channel bindings deterministically for existing configured users. Dry-run/status diagnostics must disclose actions without exposing secrets.
3. **Inbound shadow recorder:** append canonical inbound `message.created` events after existing Telegram authentication, using Telegram update/message identity for deduplication. Existing handler behavior remains unchanged.
4. **Outbound shadow recorder:** append the final assistant message and explicit delivery observations without changing streaming or notification behavior.
5. **Parity and replay diagnostic:** rebuild the channel projection from events, compare it with current history, and expose a local read-only diagnostic endpoint or page.
6. **Authority review:** only after parity and restart tests decide whether the canonical store should become the source for transcript reads and outbound delivery.

Every PR should leave Telegram, all five harnesses, memory, schedules, GitHub notifications, and installation behavior working.

### 11.4 Slice exit criteria

- One existing Telegram conversation has a stable Workshop channel identity.
- The human author and the conversation are represented by different IDs.
- Replayed events rebuild the same canonical message projection after restart.
- Duplicate Telegram delivery does not create duplicate canonical messages.
- Existing Telegram responses and streaming are unchanged.
- JSONL history remains intact and a parity diagnostic reports divergence.
- Backend selection remains a policy choice among all five installed harnesses.
- No new production service is required.

## 12. Compatibility invariants

The migration must preserve all of the following until an explicit replacement is tested and approved:

- `make config`, `make install`, `make install-status`, and protected-install ownership rules;
- Telegram private-chat interaction and streaming behavior;
- notification delivery to configured Telegram groups;
- TOTP behavior, including installations where TOTP is disabled;
- per-user backend/model selection through protected configuration and runtime settings;
- all five installed backend harnesses;
- per-user OS execution and backend authentication;
- project workspace selection and access controls;
- semantic memory and transcript provenance;
- scheduled jobs and condition monitors;
- GitHub notifications, PR review, and issue triage;
- generic webhook callers;
- file, image, voice, and service-proxy behavior;
- principal-bound internal API scopes and separated external webhook secrets.

## 13. Explicitly deferred work

The following should not be bundled into the canonical conversation foundation:

- React/Tauri production UI;
- PostgreSQL migration;
- Redis or a separate message broker;
- container or Kubernetes runtime extraction;
- root-owned backend package management;
- multi-workshop administration;
- full workflow engine;
- canvases, reactions, voice huddles, or mobile packaging;
- generalized tool-call or approval normalization before driver evidence exists;
- replacing JSONL history, Qdrant, current file storage, or APScheduler.

## 14. Risks and controls

| Risk | Control |
|---|---|
| A shadow event log silently diverges from JSONL | Add deterministic parity diagnostics and make divergence visible in tests/status before any read cutover. |
| Telegram identity leaks into the new schema | Permit Telegram IDs only in external identity/binding records and idempotency metadata. |
| The first schema overfits private chat | Model author principal, channel, agent, and project workspace separately even when the bootstrap produces one of each. |
| Event vocabulary overfits one harness | Keep the first vocabulary collaboration-only; derive run/tool capabilities from all five drivers later. |
| Dual writes create partial state | Treat shadow records as non-authoritative until command/event/outbox transactions and recovery behavior are specified. |
| Workshop work destabilizes the installed service | Use additive migrations, disabled-by-default wiring where practical, install dry runs/status checks, and a live Telegram smoke test after each deployment-affecting PR. |
| Infrastructure scope delays the desktop goal | Keep the first store in SQLite and defer runtime/storage distribution until the vertical slice proves the domain. |

## 15. Changes needed in the original design document

The design remains directionally sound, with these updates:

1. Rename the product from **Kai Workspace** to **Kai Workshop**.
2. Reserve **workspace** for project/filesystem execution contexts.
3. Rename the original top-level Workspace domain object to **Workshop**.
4. Replace the "first harness, then Codex and Claude Code" sequence. Kai already has five production-tested harness adapters; the target contract must be validated against all five.
5. Recognize the current protected backend registry and `local_process` runtime as migration foundations.
6. Move Harness Driver and Runtime Backend extraction behind the canonical conversation/event foundation unless a specific implementation blocker proves otherwise.
7. Add the existing durable Telegram queue as the model for command acceptance, while keeping commands, events, projections, and delivery outbox conceptually separate.
8. Make SQLite the explicit first-slice store and retain PostgreSQL as the scale-out target.

## 16. Phase 0 conclusion

Kai Workshop should begin by giving the conversation durable, transport-independent identity. Kai already runs agents on the server, already supports five harnesses, and already persists enough operational state to migrate safely. Building a desktop shell before defining that shared truth would force the UI to reproduce Telegram assumptions.

The next safe implementation step, after review of this map, is a test-first PR containing only the Workshop domain identifiers, versioned event envelope, SQLite event-store schema, projection contract, and idempotency tests. It should not yet alter a production message path.

## 17. Transition retirement phase

Workshop migration must finish with one authoritative implementation for each responsibility. Shadow stores, dual writes, legacy fallbacks, compatibility flags, and transport-shaped domain paths are temporary migration tools, not acceptable permanent architecture.

Transition retirement is a cross-cutting phase. It begins with the first Workshop PR, because every temporary mechanism needs an exit plan when it is introduced, and ends only after the final obsolete path has been removed. A Workshop capability is not complete merely because the new path works; it is complete when the replaced path is no longer active in production.

### 17.1 Required transition retirement ledger

The ledger is maintained in this document until the project adopts a dedicated architecture-decision or migration-tracking system. Every transitional mechanism must record:

- the mechanism and the reason it exists;
- the new authoritative component that replaces it;
- whether it is shadowing, dual-writing, dual-reading, falling back, or adapting;
- the parity, recovery, and live-operation evidence required for cutover;
- the exact condition that permits removal;
- the code, configuration, schema, diagnostics, and tests to delete or convert;
- the target migration phase or release, once one can be named responsibly.

No entry may use an indefinite condition such as "keep for compatibility." A genuinely supported compatibility surface must instead be documented as a product contract with an owner and tests; it is no longer transitional.

| Transitional mechanism | Replacement authority | Required evidence and removal gate | Retirement work | State |
|---|---|---|---|---|
| Authenticated plain-text/photo/document/voice and successful assistant-result canonical shadow writes beside current history, plus non-authoritative Telegram delivery observations | Workshop event store, message/artifact projections, and durable delivery outbox | Deterministic replay, duplicate-ingress/result tests, restart tests, full media-ingress coverage, delivery parity diagnostics, and sustained parity with current history and Telegram outcomes | Remove the `workshop_inbound_recorder`, `workshop_artifact_recorder`, `workshop_outbound_recorder`, and `workshop_delivery_recorder` bot-data adapters and their fail-open handler branches; remove the transitional `workshop_message_shadowed` JSONL marker; remove `workshop_message_parity_status` and its install-status output after canonical reads and outbox delivery become authoritative; make canonical command/event transactions and the delivery outbox the sole write and delivery paths | Active (private text finalization is authoritative in the outbox; inbound/media artifacts, remaining assistant results and delivery observations, JSONL history, and parity diagnostics remain transitional) |
| JSONL transcript writes and reads | Canonical message projection, with an explicit export facility if still useful | Canonical reads serve complete context and transcript views; migration/parity diagnostics report no unexplained divergence | Stop JSONL writes; remove production reads and dual-write recovery code; retain only a documented importer/exporter if required | Planned |
| Telegram `chat_id` used as internal identity, namespace, and routing key | Durable principal, channel, agent, and binding IDs | Private chats, notification-only groups, duplicate updates, and restart routing all resolve correctly through bindings | Confine Telegram IDs to external identity, transport binding, and idempotency records; remove chat-shaped domain keys | Active (authoritative private text now enters execution by canonical message ID, but the compatibility resolver still derives the current pool key from the human principal's protected Telegram identity; settings, locks, history, files, memory, and excluded routes remain chat-keyed) |
| `SubprocessPool` keyed by Telegram chat ID | Durable channel/agent session plus run and attempt orchestration | All five harnesses pass continuity, restart, cancellation, and isolation tests through durable identities | Remove chat-key compatibility lookup and move lifecycle ownership behind the orchestrator/runtime contract | Active (a canonical conversation-run service hides the private-text pool lookup behind a temporary compatibility resolution; the pool and all five harness processes remain keyed by that resolved integer) |
| Direct backend invocation from Telegram handlers | Transport-neutral command and run services | Telegram and the first Workshop client produce equivalent authorized runs and visible results | Remove handler-owned orchestration; leave authentication, parsing, and rendering in the Telegram adapter | Active (authoritative private text invokes through the canonical conversation-run service using only its inbound `MessageId`; a transport-neutral durable run lifecycle exists production-unused; the run-authority review holds live lifecycle activation until execution attempts, cancellation intent, atomic terminal results, and terminal replay suppression exist) |
| Direct Telegram delivery from handlers, schedules, and webhooks | Durable delivery outbox and Telegram delivery adapter | Delivery outcome events preserve binding identity; retry, crash recovery, ordering, private-chat and notification-group delivery tests pass; live delivery is verified | Migrate each remaining transport path separately; the private-text direct fallback is retired | Active (authenticated private-chat text with voice mode off uses atomic streaming finalization and a supervised exact-epoch worker and now fails closed rather than demoting to direct/shadow delivery; commands, media, voice, schedules, webhooks, files, and groups retain existing delivery) |
| Operator-invoked Workshop delivery qualification CLI | Installed evidence followed by the production delivery worker | A configured direct-chat reply is prepared without sending, survives a service restart, recovers an intentionally abandoned lease, reaches Telegram once through the exact selected delivery, and records a terminal binding-aware outcome; a configured notification group resolves through its outbound-only canonical channel, receives one atomically prepared qualification message through the exact selected delivery, and does not become an inbound conversation | Remove the qualification command and its explicit-claim-only surface after the production worker has equivalent installed restart/recovery evidence and direct delivery is retired | Active (the installed direct-chat recovery and notification-group delivery gates passed on 2026-08-12; retain until equivalent production-worker evidence exists, while the command remains unregistered and incapable of draining unrelated work) |
| Conversation-delivery authority epochs | A single durable delivery authority after direct-send rollback is retired | Activation/deactivation, restart, historical-row isolation, exact-epoch worker ownership, aggregate diagnostics, and installed rollback/reactivation evidence pass; no supported rollback crosses the direct-send/outbox boundary | Remove epoch stamping, activation/deactivation state, exact-epoch claim filters, transitional readiness output, schema columns/tables where safely migratable, and their compatibility tests | Active (production startup resumes or creates the exact epoch before ingress; installed private-text streaming, all-send, fragmentation, restart/no-replay, aggregate authority, and parity checks passed on 2026-08-12) |
| Schedule firing directly into the pool or Telegram | Durable Workshop run creation | Scheduled definitions and executions survive restart and expose run/attempt state without duplicate work | Remove schedule-specific execution path; retain schedules only as authenticated run triggers | Planned |
| GitHub and generic webhook paths that route directly to Telegram or the pool | Canonical integration commands/events plus delivery/run services | Existing GitHub group notifications, generic callers, deduplication, and secret separation pass end-to-end tests | Remove direct routing while retaining verified webhook adapters and supported external contracts | Planned |
| Per-chat settings, files, memory references, and project selection | Principal/channel/agent/project-workspace records and artifact metadata | Existing per-user isolation, context assembly, memory provenance, file delivery, and workspace access remain equivalent | Migrate namespaces and remove Telegram-derived ownership from domain storage; retire recorded compatibility `storage_path` values when an authoritative artifact store replaces local per-chat files | Active (artifact metadata foundation and photo/document/voice shadow) |
| `users.yaml` values that represent mutable application state | Workshop administration and durable policy records | Workshop administration can safely inspect and change the relevant state, and bootstrap/recovery behavior is proven | Retain only protected installation/bootstrap policy; remove duplicated mutable state and precedence rules | Planned |
| Backend-specific execution orchestration embedded in the control plane | Equal Harness Driver contracts and a Runtime Backend boundary | Capability and lifecycle tests cover Claude, Codex, Goose, OpenCode, and Pi without privileging one driver | Delete duplicated process orchestration only after the shared contracts carry the required evidence | Planned |
| Trusted-host `local_process` execution and its executable-trust warnings | Isolated Workshop workers | Worker identity, filesystem grants, credentials, networking, artifacts, cancellation, upgrade, and recovery are verified for all five harnesses | Retire trusted-host mode or explicitly reclassify it as a supported development mode; remove production mitigations that it alone requires | Planned |
| Temporary feature flags, legacy fallbacks, and compatibility diagnostics introduced during migration | The corresponding authoritative Workshop path | The owning ledger row has passed its removal gate and rollback no longer depends on the old path | Delete flags, fallback branches, stale status output, environment variables, fixtures, and tests that only preserve the retired path | Ongoing rule |
| Superseded database tables and columns | Current Workshop schema and projections | Data migration is verified, no supported binary reads the old schema, and rollback/archive policy is explicit | Freeze writes, migrate or archive data, then remove obsolete schema in a dedicated migration | Planned |

This ledger must be refined as implementation exposes concrete file names, table names, feature flags, and measurable thresholds. New rows are required whenever a PR creates another temporary path.

### 17.2 Cutover and deletion sequence

Each ledger entry follows the same controlled sequence:

1. **Introduce:** add the new path without changing production authority.
2. **Observe:** shadow or project authoritative activity and measure deterministic parity.
3. **Cut over:** make the new path authoritative, with the old path read-only only when a bounded rollback window requires it.
4. **Verify:** exercise restart, replay, duplicate delivery, failure recovery, authorization, and live installed-system behavior.
5. **Retire:** stop old writes and reads, delete fallback behavior, and migrate or archive remaining data.
6. **Simplify:** remove compatibility configuration, diagnostics, tests, documentation, and dependencies that no longer represent supported behavior.

Cutover and retirement should normally be separate small PRs. The cutover PR proves the new authority; the retirement PR makes the architectural simplification reviewable and prevents unrelated feature work from hiding legacy behavior.

### 17.3 Retirement gates

A transitional path may be removed only when all applicable gates pass:

- deterministic replay produces the expected projection from authoritative events;
- idempotency tests cover duplicate external delivery and process restart;
- crash recovery does not require the legacy path;
- authorization and principal isolation are preserved;
- all five installed harnesses remain supported where execution is involved;
- Telegram private interaction and configured notification-group delivery still work;
- GitHub and generic webhook contracts remain valid where affected;
- migration and rollback procedures are documented and tested for persistent data;
- `make config`, `make install`, and `make install-status` accurately represent the new authority;
- an installed-system smoke test confirms Kai remains usable.

Rollback must not become a reason to keep two authorities indefinitely. During a bounded rollback window, new-schema data must have an explicit backward strategy or the release must declare that rollback requires restoring a backup.

### 17.4 Final architecture after retirement

The retirement phase is complete when:

- the Workshop event store and projections are the canonical collaboration state;
- Telegram, web, and desktop are transport adapters rather than domain authorities;
- human principals, channels, agents, project workspaces, runs, attempts, artifacts, and deliveries use durable transport-independent identities;
- schedules and integrations create the same authenticated commands and runs as interactive clients;
- external delivery is durable and observable through an outbox;
- all five harnesses remain equal drivers behind common capability and lifecycle contracts;
- isolated workers provide the supported production execution boundary;
- protected files contain secrets and installation/bootstrap policy rather than duplicated mutable application state;
- Telegram IDs, provider session IDs, executable paths, and filesystem paths appear only at their appropriate adapter or runtime boundaries;
- no silent legacy fallback, dual authority, or mitigation-only configuration remains.

Any compatibility code remaining at that point must be an intentional, documented, tested product surface. Everything else is transition debt and blocks completion of the Workshop migration.

## 18. First canonical conversation authority review

**Review date:** 2026-08-11

**Scope:** Whether the canonical Workshop store may replace JSONL transcript reads or direct Telegram delivery after the first conversation-shadow sequence.

**Decision:** **Hold both authority changes.** The canonical store remains a non-authoritative shadow. JSONL remains the transcript authority, and existing Telegram delivery remains the external-effect authority.

The review distinguishes implemented structural evidence from live operational evidence. Passing deterministic tests is necessary, but it is not equivalent to proving sustained parity in the installed system.

| Gate | Evidence | Result |
|---|---|---|
| Durable typed identities, event envelope, append/idempotency contract, additive schema, and deterministic projection replay | Foundation and replay contracts introduced in the first Workshop PR; projection rebuild tests compare complete message and delivery projections | Pass |
| Deterministic default workshop, principal, agent, channel, and Telegram binding bootstrap | Bootstrap is idempotent across separate databases and service restarts; status output is non-secret | Pass |
| Authenticated plain-text Telegram ingress shadowing | Binding resolution separates sender principal from channel identity; duplicate update/message delivery is idempotent, including after restart | Pass |
| Successful assistant-result and delivery-observation shadowing | Assistant replies use the canonical channel and agent principal; result retries and delivery observations are idempotent and replayable | Pass as observation only |
| Event-to-projection and canonical-to-JSONL parity diagnostics | `make install-status` replays relevant event facts, detects projection and two-way eligible-history divergence, and exposes counts without content or identities | Pass in automated tests |
| Installed-system parity across fresh messages and a service restart | Clean installed smoke observations exist, including the post-outbox installation at `canonical=32`, `projected=32`, `replay mismatches=0`, and two-way JSONL parity, but the diagnostic has not yet accumulated reviewed, sustained clean observations from the deployed installation | Not yet proven |
| Media ingress and artifact provenance | Canonical artifact identity, additive metadata storage, deterministic projection replay, and fail-open Telegram photo/document/voice adapters record content hash, size, normalized MIME type, owning message/channel/principal, non-retrievable transport unique ID, bounded original filename where applicable, and local storage provenance. Successful voice transcription preserves the original Ogg/Opus bytes without adding their path to the backend prompt. Installed photo/document access and voice transcription were verified, followed by a service restart and clean canonical-to-JSONL parity (`canonical=30`, `projected=30`, `replay mismatches=0`). | Pass in automated and installed smoke tests |
| Canonical timeline query and client synchronization | A production-unused canonical query, concrete channel policy, versioned read-only HTTP contract, durable hashed-token human sessions, a short-lived single-use enrollment boundary, and a production-unregistered redemption HTTP contract now define authorized channel reads and secure initial device binding; production route registration and a resumable event stream do not yet exist | Partial |
| Durable outbound delivery | A production-unused foundation atomically records `delivery.requested` with pending work, uses exclusive expiring leases and immutable attempt records, applies bounded deterministic retry policy, rejects stale completion, and recovers expired work after restart. Terminal outbox settlement atomically appends versioned `delivery.succeeded` or `delivery.failed` facts carrying the exact channel-binding identity; legacy shadow observations remain replayable beside those facts. Claims are FIFO across all modes for one canonical binding, including while a predecessor is leased or waiting to retry, while distinct bindings progress independently. An unregistered Telegram text adapter/worker claims only matching transport/mode work, targets canonical private/group bindings, preserves Kai's Markdown fallback, and sanitizes provider failures. Long text receives an immutable durable fragment plan; confirmed fragment IDs survive restart and are skipped on resume. A fragment is marked `sending` before the Bot API call. Timeout, network ambiguity, invalid success evidence, or lease expiry in that window produces terminal `uncertain` state instead of an automatic duplicate. Tests cover concurrent claims, request and outcome rollback, retry exhaustion, restart/resume recovery, target isolation, notification groups, binding-aware projection, cross-delivery and fragment ordering, ambiguity, and projection rebuild preservation. | Partial (outcome, ordering, fragment, and ambiguity contracts implemented but worker remains unregistered; no production enqueue, installed live recovery evidence, or authority change) |
| Legacy transcript consumers | Context assembly, memory provenance, evaluation, and operator history tools still read JSONL directly | Not migrated |

### 18.1 Authority consequences

- Canonical shadow failures remain fail-open and must not affect Telegram responses.
- Production context assembly and transcript views must continue to read JSONL.
- Direct Telegram sends must continue to own delivery behavior for private chats, schedules, GitHub notifications, generic webhooks, files, and voice.
- `message.created` and delivery observation events are evidence, not permission to retry an external effect.
- No rollback window begins because no authority has moved.

### 18.2 Next bounded sequence

1. **Deploy and observe:** install the merged shadow sequence, create fresh plain-text exchanges, run `make install-status` before and after a service restart, and retain only counts/state as evidence. A single clean sample is a smoke test, not sustained parity.
2. **Canonical timeline query contract, unused by production:** implemented as a transport-independent, paginated channel timeline reader with stable cursors and mandatory principal/channel authorization. It accepts no Telegram IDs and replaces no JSONL reads.
3. **Canonical channel authorization policy:** implemented with explicit, replayable channel memberships. Workshop membership alone does not imply access to every direct channel; the current shared default workshop makes that unsafe. The concrete timeline authorizer requires both an explicit channel membership and membership in the channel's workshop.
4. **Authenticated read-only diagnostic surface:** the production-unregistered HTTP contract now exposes canonical, snapshot-stable timeline pages through an injected human-client authenticator. A production-unused session foundation issues hashed high-entropy tokens only to canonical humans on tracked devices, enforces bounded expiry, records last activity, and supports durable session- and device-level revocation. The enrollment boundary gives a trusted server/operator a short-lived, single-use grant bound to an existing human principal; redemption accepts no principal claim and atomically creates exactly one tracked device and session. Its production-unregistered HTTP contract accepts only the opaque grant and a device display name, returns the session token once, exposes no grant-issuance method, and gives malformed, expired, revoked, and reused grants the same bounded denial. Only hashes are retained, grant reuse and expiry fail closed, and grant/device/session revocation remain distinct. It reuses principal/channel authorization, reveals no cross-channel data, and accepts no transport identity. Backend-process credentials remain a separate trust domain. Production registration remains a separate decision after rate limits, secure transport assumptions, and installed restart/revocation evidence are defined.
5. **Media and artifact shadow:** implemented for successful Telegram photos, documents, and voice messages. The canonical artifact foundation records content identity and bounded local-storage provenance only for an existing human-authored canonical message. Each adapter marks only confirmed canonical message writes for legacy parity and fails open without changing prompts, agent invocation, JSONL authority, or Telegram delivery. Document metadata uses a normalized valid transport MIME type, a deterministic fallback, and a bounded cross-platform basename. Successful voice transcription preserves the original Ogg/Opus bytes in the existing per-user upload boundary without exposing the path to the backend prompt.
6. **Durable delivery outbox:** the production-unused foundation defines atomic `delivery.requested` work, immutable attempts, exclusive expiring claims, bounded retry/failure policy, stale-lease rejection, and crash recovery. Terminal settlement atomically records a binding-aware version-2 success or failure fact, while legacy version-1 shadow observations remain replayable. Claiming now serializes all modes by request position for one binding without blocking a different binding. Its unregistered Telegram adapter/worker handles durably planned, ordered text fragments for canonical private or notification-group bindings and claims no other transport/mode. Confirmed fragments resume without resend. The external-send/database-commit ambiguity is fail-closed: an in-flight fragment without confirmation becomes terminally uncertain and requires reconciliation rather than automatic retry. It neither registers at startup nor enqueues or replaces a direct Telegram send. An operator-invoked qualification command can prepare the latest existing canonical Kai reply for one configured direct Telegram user, inspect it, deliberately abandon an exact lease, and run only that named delivery. It cannot drain or recover unrelated work and does not create another message or alter transcript parity. Installed live recovery evidence remains required before activation.
7. **Repeat the authority review:** require sustained installed parity, media coverage, read-client restart/reconnect evidence, and live delivery recovery before approving a cutover PR.

The secure client-enrollment boundary and its isolated HTTP redemption contract for item 4 are now defined but remain production-unused. They bind the initial device and session to a principal established by the server or operator, never by an untrusted client identity claim, and preserve the separation from backend-process credentials. Session, device, and enrollment rows are mutable security state rather than replayed collaboration projections; later append-only audit events may record their lifecycle but must never be capable of reactivating a credential.

Production client registration is additionally blocked on [#835](https://github.com/dcellison/kai/issues/835): canonical collaboration projection rebuilds must preserve mutable device, session, grant, and revocation state across service restart.

Successful installed voice ingress, service restart, and clean canonical-to-JSONL parity are now recorded, including a clean post-outbox install at 32 canonical/projected messages. Binding-aware terminal facts and per-binding FIFO claims are implemented without startup registration or production enqueue. The deliberately invoked installed qualification path is now available to collect real Telegram delivery and restart-recovery evidence without making the outbox the production authority; that evidence must be collected after deployment before another authority review. Operator-facing grant issuance and production route registration remain separate, explicit decisions requiring rate limits, secure transport assumptions, and installed restart/revocation evidence.

## 19. Second canonical conversation authority review

**Review date:** 2026-08-12

**Scope:** Whether installed parity and the live Telegram outbox qualification are sufficient to register the delivery worker or replace any direct Telegram delivery path.

**Decision:** **Hold the delivery cutover.** The outbox's installed recovery mechanics are proven, but production message creation and delivery enqueue do not yet form one authoritative transaction. Registering the worker before that boundary exists would only drain manually or partially prepared work; it would not make normal delivery durable end to end. Existing direct Telegram delivery therefore remains authoritative, and the worker remains unregistered.

This decision does not reopen the mechanics already qualified. It separates a successful worker/recovery proof from the still-missing command-to-delivery authority boundary.

| Gate | Evidence | Result |
|---|---|---|
| Sustained canonical projection and JSONL parity | Installed observations remained clean through plain text, photo, document, voice, service restarts, outbox foundations, and the live qualification. The post-qualification diagnostic reported `canonical=36`, `projected=36`, zero replay mismatches, and two-way JSONL parity. | Pass for this review |
| Media ingress | Installed photo/document access and voice transcription succeeded, including restart verification and clean parity. | Pass |
| Durable worker mechanics | Automated contracts cover exclusive claims, bounded retry, per-binding FIFO ordering, fragment resume, stale leases, projection rebuild, and fail-closed ambiguous sends. | Pass |
| Installed direct-chat recovery | One existing canonical reply was prepared without sending, claimed and deliberately abandoned, observed as leased after a Kai restart, recovered after lease expiry, delivered once through Telegram on attempt 2 of 3, and recorded as succeeded. The extra transport delivery did not create another canonical message or disturb parity. | Pass |
| Atomic assistant result and delivery request | `record_workshop_outbound_message` currently commits the canonical assistant message independently. `WorkshopDeliveryOutbox.request_delivery` starts a separate transaction, and production handlers still send directly. A crash between those operations could leave a canonical reply without durable delivery work. | Blocker |
| Installed notification-group delivery | Canonical group binding and delivery behavior are covered by automated tests, but the installed qualification exercised only a direct chat. Existing GitHub notification-group behavior must remain unchanged. | Blocker before group cutover |
| Production worker lifecycle | The worker has a tested stop-event loop but no Kai startup/shutdown owner, readiness signal, or installed graceful-shutdown evidence. This absence is intentional until atomic enqueue exists. | Blocker |
| Production transport coverage | The Workshop adapter currently delivers text. Existing direct paths also cover command responses, schedules, GitHub and generic webhooks, files, and voice. Each may move only through an explicit bounded cutover that preserves its supported behavior. | Partial |

### 19.1 Next bounded implementation milestone

The production-unused application service atomically creates one canonical assistant reply and its Telegram text delivery request in the same SQLite transaction. It:

- resolves the existing inbound message, agent principal, channel, and canonical Telegram binding without accepting a transport identity from its caller;
- appends `message.created` and `delivery.requested`, projects the canonical message, advances the projection checkpoint across both facts, and inserts the pending outbox row under one transaction;
- uses deterministic identities and idempotency keys so replay returns the same message and delivery without duplicate work;
- rolls back both the canonical reply and delivery request when binding resolution, event append, projection, or outbox insertion fails;
- rejects ambiguous or missing bindings rather than guessing a destination;
- remains unused by production handlers and leaves the worker unregistered.

Rollback tests cover message projection failure and outbox insertion failure; both leave no assistant event, projected reply, delivery-request event, or pending work. Restart and concurrent-connection retries produce one deterministic message and one delivery, changed content fails closed, and a missing or ambiguous canonical Telegram binding is rejected without accepting a destination from the caller. A pre-existing message-only half-state is not silently repaired.

The service remains production-unused. Next, separately qualify installed notification-group delivery, define lifecycle ownership, and conduct another explicit cutover review. No temporary delivery feature flag is introduced by this sequence.

### 19.2 Installed notification-group qualification gate

Effective negative Telegram notification destinations now bootstrap as distinct canonical `notification` channels. They reuse configured human principals and attach Kai, but the inbound recorder rejects the channel kind so an outbound destination cannot silently become an interactive group. This bootstrap is additive and does not change GitHub routing.

The operator-only qualification command atomically creates one recognizable Kai-authored message in the selected notification channel and its pending text delivery. Preparation is deterministic and idempotent, accepts only a configured canonical negative Telegram destination, does not send, and cannot claim unrelated work. The existing exact-delivery `status`, `simulate-interruption`, and `run` actions remain the only ways to exercise it. Direct-chat JSONL parity excludes outbound-only notification channels while projection replay still covers their canonical facts.

Installed evidence passed on 2026-08-12. The deployed bootstrap reported three Telegram bindings and six explicit channel memberships for two humans. Delivery `dlv_81b5d54eb64355eaad951e21105411fd` was prepared without sending, then delivered through the exact qualification action on attempt 1 of 3 and reached the configured Telegram group with the recognizable qualification text. The subsequent diagnostic remained clean at `canonical=39`, `projected=39`, zero replay mismatches, `JSONL matched=38`, zero missing or unmatched records, and one direct Telegram parity channel. A redelivered GitHub event then reached the same group through the unchanged legacy route.

## 20. Third canonical conversation authority review

**Review date:** 2026-08-12

**Scope:** Whether atomic assistant-result enqueue plus installed direct-chat and notification-group evidence are sufficient to register the Workshop Telegram worker or replace a production delivery path.

**Decision:** **Hold worker registration and production routing cutover.** The canonical message/outbox transaction and both installed Telegram targets are now proven. The remaining blocker is ownership of the worker inside Kai's startup, readiness, fault, and graceful-shutdown lifecycle. Registering a bare task before that contract exists would make delivery authority depend on implicit task behavior and could conceal a dead worker behind an otherwise healthy service.

This decision also keeps transport coverage bounded. Direct Telegram sends remain authoritative for normal replies, schedules, GitHub and generic webhooks, files, and voice. The successful group qualification proves the Workshop transport target; it does not authorize moving GitHub routing or making a notification destination interactive.

| Gate | Evidence | Result |
|---|---|---|
| Atomic assistant result and delivery request | The production-unused application service commits `message.created`, its projection, `delivery.requested`, and pending outbox work in one SQLite transaction; rollback, idempotency, concurrency, restart, and half-state tests pass. | Pass |
| Installed direct-chat recovery | The exact direct-chat delivery survived an abandoned lease and service restart, recovered after expiry, and reached Telegram on a bounded retry without creating another canonical message. | Pass |
| Installed notification-group delivery | The effective negative destination bootstrapped as an outbound-only canonical channel. Its exact qualification delivery succeeded on attempt 1 of 3, reached Telegram, and preserved clean direct-chat parity. | Pass |
| Existing GitHub notification behavior | A GitHub webhook redelivery reached the same configured Telegram group after the Workshop qualification, proving that the unchanged legacy route still works. | Pass |
| Inbound isolation | Notification channels create no external identity and the inbound recorder rejects their channel kind. Automated tests cover the fail-closed boundary. | Pass |
| Worker lifecycle ownership | The worker loop accepts a stop event, but Kai has no explicit owner for task creation, readiness, failure propagation, shutdown ordering, or installed lifecycle evidence. | Blocker |
| Production route coverage | No production path uses atomic enqueue and the worker remains unregistered. Non-text transport modes remain on their existing direct paths. | Partial by design |

### 20.1 Next bounded implementation milestone

The production-unused Telegram delivery runtime owner now wraps the existing worker. It:

- creates at most one worker task and exposes an explicit ready state;
- recovers expired leases before reporting ready;
- surfaces an unexpected worker exit or exception instead of silently leaving Kai healthy;
- stops new polling during shutdown and awaits the worker's current serialized iteration without cancelling an in-flight external send;
- makes repeated or out-of-order start/stop calls deterministic;
- remains uncalled by `main.py`, enqueues no production work, and preserves all direct Telegram behavior.

Automated contracts cover clean startup, recovery-before-ready ordering, duplicate start rejection, idle and active graceful shutdown, fault propagation, and the absence of any worker task when the owner is never explicitly started. No production startup or routing code imports or starts the owner.

### 20.2 Next bounded review

Conduct a focused cutover review for one normal direct-chat text-reply path. The review must define how Kai's application lifecycle will supervise the owner and how one reply path will switch atomically from canonical result creation to durable delivery without double-sending. GitHub notification routing, schedules, files, and voice remain separate later cutovers.

## 21. Fourth canonical conversation authority review

**Review date:** 2026-08-12

**Scope:** Whether the explicit Telegram delivery runtime owner is sufficient to cut one normal direct-chat plain-text reply path over to canonical atomic enqueue and the Workshop outbox.

**Decision:** **Hold the production cutover while implementing one production-unused streaming-finalization boundary.** The lifecycle owner clears the previous task-ownership blocker, but starting it or replacing the current final send now could duplicate a normal reply or deliver stale qualification work. The next work is a bounded delivery-adapter foundation, not a broad handler or lifecycle refactor.

The normal reply path is not a single final `send_message` call. `_handle_response` may create a Telegram message from a stable streaming prefix, edit it repeatedly, and then either edit that same message to the final text or send chunked continuation messages. In some exchanges the last streaming edit already equals the terminal agent response, so the current final branch deliberately performs no further Bot API call. The existing Workshop worker can only send new final-text fragments. Enqueuing the same final response without accounting for that live message would therefore produce a second Telegram copy.

| Gate | Evidence | Result |
|---|---|---|
| Atomic canonical reply and delivery request | `record_outbound_message_with_delivery` resolves the canonical inbound message and Telegram binding, commits the assistant message and pending delivery together, and has rollback, idempotency, restart, and concurrency coverage. | Pass, production-unused |
| Delivery worker lifecycle owner | The owner recovers leases before readiness, owns one task, exposes unexpected termination, and cooperatively awaits active work without cancellation. Focused and full-suite tests pass. | Pass in automated tests; intentionally not installed-active |
| Plain-text route isolation | `handle_message` is a distinct authenticated ingress, but `_handle_response` is also shared by photo, document, and voice paths and contains text, voice-only, and text-plus-voice delivery modes. | Partial; cutover eligibility must be an explicit typed input from the caller, never inferred from prompt shape or backend behavior |
| Streaming finalization | A live Telegram message may already contain all or part of the final answer before canonical enqueue. The current outbox has no durable edit target or edit-fragment operation. | Blocker; a send-only worker would duplicate or abandon the live response |
| Historical-work isolation | The worker currently claims every eligible Telegram text delivery. Existing qualification rows are intentionally production-unused but are not classified separately from future conversation replies. | Blocker; first production startup must not drain historical qualification work |
| Database ownership | Canonical handler writes use Kai's initialized SQLite connection under the Workshop event lock. A continuously polling worker on that same connection could interleave transaction boundaries with handler work. | Blocker; the runtime needs its own opened Workshop store/connection and must close it after the worker stops |
| Main-loop supervision and shutdown order | `main.py` currently blocks on an unrelated never-set event. The runtime's `wait()` can instead expose worker death to the existing fatal top-level error path. | Design ready, not wired; start must finish before updates are accepted, and shutdown must stop the worker before closing its store or Telegram client |
| JSONL compatibility | JSONL remains the transcript/context authority and is written before current final delivery. Canonical-to-JSONL parity remains observable. | Preserve for this delivery cutover; transcript authority is a separate later review |
| Rollback and old work | Disabling a worker while committed conversation deliveries remain non-terminal could cause a later reactivation to send an old reply. | Blocker; activation and rollback require explicit work classification and reconciliation diagnostics |

### 21.1 Authority consequences

- The runtime owner remains unstarted and direct Telegram delivery remains authoritative.
- Installing the runtime-owner code does not exercise or qualify production lifecycle wiring.
- Normal reply streaming, commands, media, voice, schedules, GitHub and generic webhooks, files, and notification-group routing remain unchanged.
- A handler must never both enqueue an authoritative final reply and invoke the legacy final-send branch.
- Worker failure after activation must make Kai unhealthy and trigger the existing process supervisor; it must not leave a loaded service silently unable to deliver.
- Existing qualification work must not become eligible merely because the production worker starts.

### 21.2 Next bounded implementation milestone

Add a production-unused durable Telegram streaming-finalization contract. It should:

1. classify delivery work by durable purpose so a future production worker can claim normal conversation replies without draining direct-chat or notification-group qualification rows;
2. bind a confirmed streaming-preview Telegram message ID to the canonical direct channel and inbound message without accepting a chat destination from the caller;
3. let atomic assistant-result enqueue resolve that preview internally and create an immutable fragment plan whose first operation edits the preview and whose remaining operations send continuation fragments, or whose operations all send when no preview exists;
4. keep streaming previews explicitly non-final so the authoritative finalization always has a distinct operation, including when the backend's last streaming snapshot equals its terminal response;
5. classify edit failures and external-effect ambiguity as rigorously as send failures, retaining fail-closed reconciliation rather than automatic duplicate delivery;
6. remain uncalled by production handlers and leave the runtime unregistered.

Tests must cover a short streamed reply, a streamed reply whose final snapshot was already published, a long fragmented reply, a response with no preview, a deleted or uneditable preview, restart after confirmed fragment progress, send/edit ambiguity, qualification-work exclusion, and binding isolation.

### 21.3 Later cutover contract

Only after the production-unused finalization contract passes another review should one cutover PR:

- select Workshop delivery explicitly in `handle_message` only for an authenticated plain-text direct message with a confirmed canonical inbound ID and voice mode off;
- retain the legacy path for commands, photo, document, voice, text-plus-voice, voice-only, and any ingress that could not be canonically accepted;
- expose a locked `sessions` adapter for atomic reply-plus-delivery creation instead of giving the handler a raw store;
- construct the worker on a dedicated Workshop store connection, finish recovery before accepting Telegram updates, supervise `runtime.wait()` as the service lifetime, and stop the runtime before closing that store or the Telegram application;
- skip the legacy final send only after authoritative enqueue succeeds, and fail closed with a bounded user-visible operational error when commit outcome is uncertain;
- report classified pending, leased, retrying, failed, and uncertain conversation work through non-secret installed diagnostics;
- document rollback as: stop the worker, reconcile all non-terminal conversation deliveries, restore the legacy route, and prove no old delivery can replay on later activation.

Installed qualification must then demonstrate one streamed short reply finalized in place with no second copy, one fragmented long reply in order, a restart between enqueue and delivery, cooperative shutdown during active work, fatal recovery from an injected worker fault, clean canonical/JSONL parity, and unchanged media, voice, GitHub notification-group, schedule, file, and command behavior.

### 21.4 Durable delivery-purpose foundation

The first streaming-finalization prerequisite is now implemented without production registration. Every new delivery request must identify one durable purpose: `conversation_reply` or `qualification`. Existing version-10 outbox rows migrate to `qualification`, which is the fail-safe classification for all delivery work created before normal conversation routing exists.

Claims, expired-lease recovery, and per-binding ordering are purpose-scoped. A qualification row therefore cannot be claimed, recovered, or used as an ordering predecessor by a conversation worker, and the Telegram worker instance owns exactly one purpose. The installed qualification command is fixed to `qualification`; the production-unused atomic assistant-result service is fixed to `conversation_reply`. Neither accepts a purpose from an external caller.

Migration, conflict, lane-isolation, recovery-isolation, and worker-exclusion tests pin this boundary. The runtime remains unregistered and normal Telegram delivery remains unchanged. The next bounded slice is canonical binding of a confirmed streaming preview to its direct channel and inbound message; it must remain production-unused and must not yet edit or send through the Workshop worker.

### 21.5 Durable streaming-preview binding foundation

The second streaming-finalization prerequisite is now implemented without a
production caller. A confirmed Telegram streaming preview can be durably bound
to one canonical human-authored inbound message in a direct channel. The input
contract contains only the typed inbound message ID, the positive Telegram
message ID returned by the already-confirmed send, and its confirmation time.
It accepts no chat destination, channel ID, binding ID, transport, or external
channel identity.

The service resolves the direct channel and its unique Telegram binding inside
the write transaction. Missing, non-direct, assistant-authored, or ambiguous
targets fail closed. One inbound message cannot change preview identity, and
one Telegram message cannot cross canonical inbound-message boundaries. The
persisted state is explicitly `confirmed_non_final`, so a streaming snapshot is
never mistaken for authoritative completion merely because its text happens to
match the terminal response.

The binding is operational external-effect state rather than a replayed
conversation projection. It therefore survives process restart and canonical
projection rebuild, while idempotent and concurrent retries produce only one
record. Migration, destination-isolation, conflict, rollback, restart, and
rebuild tests pin this boundary. No production handler calls it, and it does
not send or edit Telegram.

The next bounded slice is production-unused atomic assistant finalization. It
must resolve any preview internally and build an immutable operation plan whose
first operation edits that confirmed preview and whose remaining operations
send continuation fragments. When no preview exists, every operation must be a
send. It must still leave production routing and the runtime registration
unchanged.

### 21.6 Atomic streaming-finalization plan foundation

The production-unused atomic finalization boundary is now implemented. Schema
version 13 gives every delivery a durable execution contract and every fragment
an immutable `send` or `edit` operation with an optional validated existing
Telegram message target. Existing outbox work migrates to `send_fragments`, and
existing fragments migrate to `send`, preserving the behavior and eligibility
of every pre-migration row.

Given only a canonical inbound message ID and the terminal assistant text, the
application service resolves the agent, direct channel, Telegram binding, and
optional confirmed preview internally. In one SQLite transaction it creates or
reuses the canonical assistant reply, creates its `conversation_reply`
delivery under the `streaming_finalization` execution contract, and persists
the complete immutable operation plan. A confirmed preview produces one edit
of its exact Telegram message followed by zero or more continuation sends. No
preview produces only sends. The edit remains explicit even when the last
streaming snapshot already equals the terminal response.

The transaction fails closed on routing disagreement, changed content, changed
plan semantics, or any message/delivery/plan half-state. Insert failure at any
stage rolls back all newly created state. Restart and later observation time do
not change the deterministic reply, delivery, or operation plan.

This foundation cannot yet make an external effect. The current Telegram
worker explicitly claims only `send_fragments` work, and its adapter rejects an
edit operation before calling Telegram. Claims and expired-lease recovery are
execution-contract scoped, so the send-only worker cannot drain or mutate a
streaming-finalization plan. No production handler calls the atomic service and
the runtime remains unregistered.

The next bounded slice is a production-unused, edit-capable Telegram
finalization adapter and worker. It must execute only the
`streaming_finalization` contract, preserve ordered confirmed progress across
restart, apply the established Markdown fallback to both sends and edits, and
classify deleted/uneditable targets and ambiguous edit/send outcomes without
automatic duplicate delivery. It must remain unregistered until another
explicit cutover review.

### 21.7 Streaming-finalization execution foundation

The production-unused Telegram finalization adapter and worker now execute the
immutable operation plan without widening the existing send-only worker. The
new worker is fixed to `conversation_reply`, `streaming_finalization`, Telegram,
and text work. It cannot claim qualification rows or `send_fragments` work; the
existing worker remains unable to claim streaming-finalization rows.

For an edit operation, the adapter uses the canonical binding's Telegram target
and the persisted positive preview message ID. It applies Kai's Markdown-first,
plain-text-fallback presentation behavior. Telegram's explicit "message is not
modified" response confirms that the desired terminal snapshot already exists
and completes the distinct edit operation. A deleted or uneditable target
fails permanently with a sanitized edit-specific code; it never falls back to
sending a duplicate message. Successful edit evidence must identify the exact
persisted target message.

Confirmed fragment progress is monotonic. After a successful edit, a retry or
restart skips that edit and resumes at the first pending continuation send. A
rate limit can retry only an operation proven not to have taken effect. Timeout,
network ambiguity, invalid success evidence, cancellation, or lease expiry
after an operation enters `sending` becomes terminal `uncertain` state and is
never retried automatically. Lease recovery distinguishes uncertain edits from
uncertain sends in its non-secret error classification.

Automated contracts cover exact edit targeting, Markdown fallback, an already
identical terminal snapshot, deleted/uneditable previews, mismatched success
evidence, edit and continuation-send ambiguity, short edit-only replies,
edit-plus-send fragmentation order, all-send plans, retry and restart after a
confirmed edit, crash recovery, and execution-contract and purpose isolation.

The adapter and worker remain unregistered. No production handler imports or
calls them, no live delivery route changes, and normal Telegram, media, voice,
commands, schedules, GitHub/generic webhooks, files, and notification-group
behavior remain authoritative on their existing paths.

The next bounded step is a fifth explicit cutover review. It must decide
whether the complete production-unused finalization boundary is sufficient to
wire one authenticated direct-chat plain-text path and its dedicated worker
store/runtime, or whether another installed qualification path is required
first. The review must resolve startup handling of historical conversation
work, non-secret diagnostics and reconciliation, fail-closed user-visible
errors, shutdown ordering, and rollback before authorizing any production
registration.

## 22. Fifth canonical conversation authority review

**Review date:** 2026-08-12

**Scope:** Whether the complete production-unused Telegram streaming-
finalization boundary is sufficient to activate one authenticated direct-chat
plain-text reply path and its dedicated worker store/runtime.

**Decision:** **Hold production activation while adding a durable conversation-
delivery authority epoch and aggregate readiness diagnostic.** The operation
plan, edit-capable adapter, retry and ambiguity rules, and runtime owner are
now adequate for the eventual route. The remaining risk is not Telegram
mechanics. It is proving which durable rows a newly authoritative worker owns,
both on first activation and after a rollback and later reactivation.

Purpose and execution-contract isolation prevent qualification and legacy
send-fragment rows from entering this worker. They do not distinguish a
streaming-finalization row created before production activation from one
created while the route is authoritative. Starting the worker would therefore
make every matching historical row eligible. Likewise, merely stopping the
worker and restoring direct sends would leave committed non-terminal rows that
could be delivered unexpectedly after a later restart or reactivation.

| Gate | Evidence | Result |
|---|---|---|
| Immutable finalization plan | One transaction resolves the canonical direct binding and optional confirmed preview, then records the assistant reply, delivery, and exact edit/send sequence. Idempotency, rollback, restart, fragmentation, and binding-isolation tests pass. | Pass, production-unused |
| Finalization execution | The dedicated worker claims only Telegram text `conversation_reply` work under `streaming_finalization`; confirmed operations resume monotonically and ambiguous effects become terminally uncertain. | Pass, production-unused |
| Worker lifecycle ownership | The runtime owner recovers before readiness, exposes worker death, and cooperatively drains an active iteration during shutdown. | Pass, production-unused |
| Live route eligibility | The authenticated plain-text handler is identifiable, but it shares `_handle_response` with media and voice. Eligibility must be passed explicitly with the canonical inbound ID and voice mode off. | Design ready; not wired |
| Dedicated database ownership | A worker can open and own a separate `WorkshopEventStore` connection. Startup must finish recovery before webhook or polling ingress, and shutdown must stop ingress and the runtime before closing Telegram and database resources. | Design ready; not wired |
| Historical-work ownership | Purpose and execution-contract filters do not record whether matching conversation work belongs to the current production authority period. | Blocker |
| Rollback and reactivation | Kai cannot durably deactivate one authority period, prove its work reconciled, and prevent those rows from replaying during a later period. | Blocker |
| Commit-outcome fallback | The atomic service rolls back ordinary failures, but the live adapter still needs to resolve deterministic state after a database exception. It may use the legacy final-send path only after proving that no authoritative finalization committed; an indeterminate outcome must fail closed with a bounded operational message. | Blocker for route wiring, after authority epochs |
| Operator diagnostics | `install-status` reports bootstrap and transcript parity but no classified conversation-delivery readiness, active authority period, non-terminal work, terminal failures, or uncertain effects. | Blocker |
| Installed qualification | Existing qualification proves durable Telegram send, restart recovery, and both direct and notification-group targets. Automated finalization tests prove edit semantics. Another send-only qualification would not prove handler eligibility, authority activation, or lifecycle wiring. | No additional installed qualification before the authority-epoch foundation; installed cutover evidence remains mandatory afterward |

### 22.1 Authority and compatibility consequences

- Direct Telegram delivery remains authoritative for every production path.
- No finalization runtime or dedicated worker store is registered by this
  review.
- JSONL remains the transcript/context authority during this delivery-only
  transition. A Workshop-owned text reply must not also record a legacy
  delivery observation after the later cutover.
- Commands, media, voice, text-plus-voice, schedules, GitHub and generic
  webhooks, files, and notification-group routing remain outside this cutover.
- An individual terminal worker failure may be reported by aggregate
  diagnostics without crashing Kai. Unexpected worker-loop termination must
  still make the service unhealthy.
- Diagnostic and reconciliation output must contain counts and state names
  only. It must never expose message bodies, Telegram IDs, delivery IDs, lease
  IDs, worker IDs, or provider errors.

### 22.2 Next bounded implementation milestone

Add a production-unused durable **conversation-delivery authority epoch** and
non-secret readiness diagnostic. The boundary must:

1. transactionally activate at most one authority epoch and retain its durable
   identity across ordinary service restarts;
2. require atomic streaming-finalization enqueue to resolve and stamp the
   active epoch internally, accepting no epoch or authority claim from the
   handler;
3. restrict finalization claim, lease recovery, ordering, and settlement to
   the exact active epoch, so pre-activation and previously deactivated work
   can never be drained by a later worker;
4. fail activation when unclassified or unreconciled matching historical work
   exists, without deleting, silently reclassifying, or delivering it;
5. refuse deactivation while that epoch has pending, leased, or retry-wait
   work, and retain failed or uncertain terminal evidence for explicit
   operator reconciliation;
6. expose aggregate `install-status` readiness for epoch state and classified
   pending, leased, retrying, succeeded, failed, and uncertain work without
   revealing identifiers or content;
7. keep activation, deactivation, production enqueue, worker registration, and
   direct-send replacement uncalled by `main.py` and handlers.

This epoch is transitional delivery-authority state, not a user-visible
Workshop concept. Record it in the transition register and retire it only when
all supported transports use one durable delivery authority and rollback no
longer crosses a direct-send/outbox boundary.

After this foundation, conduct a sixth explicit cutover review. That review
must verify deterministic post-error commit resolution, the locked `sessions`
adapters, startup and shutdown wiring, aggregate diagnostics, and an executable
rollback procedure before authorizing one production handler path.

### 22.3 Conversation-delivery authority-epoch foundation

The production-unused authority boundary is now implemented. Schema version 14
adds durable authority epochs and nullable epoch ownership on outbox rows.
Existing rows remain unclassified rather than being silently adopted. A first
activation fails closed while any matching unclassified conversation
finalization exists; ordinary restarts reuse the same active epoch.

Atomic streaming finalization resolves the active epoch inside its existing
transaction and stamps both the request event and outbox row. Its caller cannot
supply an epoch. The finalization worker must be constructed with one typed
epoch and scopes claim, per-binding ordering, lease recovery, and settlement to
that exact still-active epoch. A later epoch therefore cannot drain work from
a prior authority period.

Deactivation refuses pending, leased, or retry-wait work. It also requires an
explicit acknowledgement when terminal failures exist, including failures with
uncertain-fragment evidence, and retains that evidence after acknowledgement;
reactivation creates a new epoch. `install-status` reports only aggregate epoch,
classification, active-status, and uncertainty counts. It exposes no epoch,
delivery, lease, worker, Telegram, message, or provider identifiers or content.

This foundation was production-unused when introduced. Section 23 records the
subsequent cutover decision and the narrow production wiring that now consumes
it.

## 23. Sixth canonical conversation authority review and first cutover

**Review date:** 2026-08-12

**Scope:** Whether the completed authority-epoch boundary, deterministic
finalization transaction, exact-epoch worker, aggregate diagnostic, and
lifecycle owner are sufficient to replace direct final delivery for one
authenticated private-chat plain-text path.

**Decision:** **Authorize and implement the first production cutover.** The
remaining work was production assembly rather than another missing domain
foundation. The cutover is deliberately narrow:

- only `handle_message` can select it;
- the Telegram chat must be the authenticated user's private chat;
- canonical inbound recording must have succeeded;
- voice mode must be off;
- commands, groups, photos, documents, voice messages, text-plus-voice,
  voice-only, schedules, GitHub and generic webhooks, and files retain their
  existing paths.

Production startup opens a dedicated Workshop store, transactionally resumes
or creates the single conversation-delivery authority epoch, recovers its
expired leases, and starts the supervised finalization worker before webhook
or polling ingress begins. Unexpected worker exit is service-fatal. Shutdown
stops ingress, cooperatively stops the worker, closes its store, and only then
closes the Telegram application and shared session database.

The handler binds a confirmed streaming preview only after Telegram returns a
positive message ID. On successful agent completion it uses the locked session
adapter to atomically persist the canonical assistant message, exact-epoch
delivery request, and immutable edit/send plan. Once that commit is confirmed,
the handler performs no direct final send or shadow delivery observation. The
worker edits the preview or sends the planned fragments.

An SQLite error is resolved by repeating the deterministic operation while the
session write lock is still held. The retry either creates work rolled back by
the first attempt or observes the already-committed identical state. If that
resolution also fails, the outcome is classified as uncertain and the handler
refuses direct fallback, returning only a bounded operational notice. A
definite preview or finalization preparation error retains the current direct
delivery path so an isolated canonical failure does not make Kai unusable.

### 23.1 Rollback contract

Rollback must not cross authority periods:

1. stop Telegram ingress and the Kai service;
2. inspect `make install-status` and reconcile every active pending, leased,
   retrying, failed, or uncertain conversation delivery;
3. run `python -m kai workshop delivery-authority deactivate` as the deployed
   database owner, adding `--acknowledge-terminal-failures` only after reviewing
   retained terminal evidence;
4. restore the prior direct-delivery build;
5. before any later reactivation, verify that prior non-terminal and
   unacknowledged counts are zero.

Deactivation refuses non-terminal work and never deletes or reassigns rows. A
future activation creates a new epoch, so its worker cannot replay prior-epoch
work.

### 23.2 Required installed evidence

The code cutover is not considered qualified until the deployed system proves:

- startup reports an active authority epoch and clean aggregate counts;
- one short streamed reply is finalized in place with no second copy;
- one response without a preview and one fragmented response arrive once and
  in order;
- restart recovery delivers committed work without replaying confirmed
  fragments;
- media, voice, commands, GitHub notification-group delivery, schedules, and
  files remain unchanged;
- canonical projection and JSONL parity remain clean.

After that evidence, the next implementation milestone is removal of the
private-text direct fallback and its shadow-delivery compatibility branch—not
another delivery foundation.

## 24. Private-text fallback retirement

**Review date:** 2026-08-12

**Evidence:** The installed cutover produced one in-place streamed response,
one no-preview response, and one two-fragment response with a clean boundary
between numbered items 91 and 92. A service restart replayed no confirmed
fragment and the next response arrived once. The authoritative diagnostic then
reported one active epoch, four succeeded deliveries, zero pending, leased,
retrying, failed, or uncertain deliveries, no prior-epoch work, exact canonical
projection parity, and no missing or unmatched JSONL records. Existing installed
outbox qualification had already exercised abandoned-lease recovery, while the
exact-epoch conversation worker's committed-work recovery remains pinned by its
integrated restart tests. Route-selection tests and the deliberately narrow
production selector preserve previously verified command, media, voice, file,
group, schedule, and webhook behavior.

**Decision:** **Retire direct and shadow fallback for the authoritative private
text route.** Once an authenticated private-chat text request with voice mode
off selects Workshop delivery, it may not demote back to legacy delivery:

- missing canonical inbound identity or runtime adapters stop before backend
  invocation and return a bounded operational notice;
- a definite streaming-preview binding failure replaces the non-final preview
  with a bounded notice and stops;
- a definite finalization failure replaces an existing preview, or sends one
  bounded notice when no preview exists, and stops;
- an uncertain finalization result continues to refuse a resend because the
  durable commit may already exist;
- none of these branches writes an outbound shadow result, delivery observation,
  or direct copy of the agent response.

This retirement is scoped to the authoritative private-text route. The shadow
recorders and direct-delivery implementation remain transitional dependencies
for explicitly excluded routes until each receives its own authority review and
cutover.

## 25. Canonical conversation-run service boundary

**Implementation date:** 2026-08-12

The authoritative private-text path now enters backend execution through a
transport-neutral service request containing only the canonical inbound
`MessageId`. The service resolves and validates the durable human author,
channel membership, workshop, and exactly one attached agent. The Telegram
handler can no longer supply an agent identity, backend identity, model, or
pool key for this path.

The existing `SubprocessPool` still requires an integer key to preserve the
five qualified harnesses, their protected per-user configuration, session
continuity, workspace selection, and OS-user execution. A deliberately private
compatibility resolution therefore maps the canonical human principal's
protected Telegram external identity to that key. The prepared run returned to
the handler exposes only canonical IDs and normalized stream/workspace methods;
it does not expose the compatibility key. This is an adapter, not a new source
of identity authority.

The boundary is intentionally narrow:

- it is production-used only by authenticated private-chat text whose canonical
  inbound write succeeded;
- the existing private-text durable Telegram finalization and fail-closed
  delivery behavior is unchanged;
- text with voice output, media, commands, jobs, integrations, and notification
  channels retain their current execution paths;
- process locks, settings, JSONL history, memory, and the live harness pool
  remain keyed by their existing compatibility identity;
- the production conversation service does not yet create the durable run or
  attempt facts defined in sections 26 and 28; the execution authority and
  leases exist only as production-unused contracts;
- no Workshop desktop/web endpoint is registered by this change.

Automated contracts require one canonical human channel member, exactly one
attached agent, exactly one valid compatibility identity, resolver-message
identity preservation, hidden pool-key delegation, handler invocation by
canonical message ID, and unchanged private-text delivery authority.

Section 26 records the completed production-unused durable run lifecycle that
now sits behind this boundary.

## 26. Production-unused durable run lifecycle

**Implementation date:** 2026-08-12

Schema version 15 adds a canonical `RunId`, a replayed `runs` projection, and
five version-1 lifecycle facts: `run.accepted`, `run.started`, `run.completed`,
`run.failed`, and `run.cancelled`. One human-authored inbound `MessageId`
deterministically identifies at most one run. Acceptance resolves the canonical
human membership, channel, workshop, and exactly one attached agent without
requiring a Telegram identity or accepting any caller-selected execution
identity.

The version-1 lifecycle state machine was deliberately small:

```text
accepted --> started --> completed
    |           |  +--> failed
    +-----------+-----> cancelled
```

The requesting human is the recorded actor for acceptance. Version-1 events
remain replayable, but their direct transition helper was removed when schema
version 16 introduced the execution authority in section 28. New cancellation
is a human-authored request followed by an execution-authority acknowledgement;
new start and terminal facts require a fenced attempt. Terminal failures and
cancellations persist only bounded lowercase classification codes; provider
messages, prompts, output text, credentials, transport IDs, executable paths,
and harness-specific payloads are excluded. Lifecycle timestamps cannot
precede their prior canonical fact.

Deterministic event IDs and idempotency keys make acceptance repeatable.
Execution transition idempotency and arbitration now belong exclusively to the
fenced authority described in section 28. Conflicting terminal facts, invalid
ordering, unknown runs, ambiguous canonical agent attachment, actor mismatch,
malformed payloads, and time reversal fail closed. The canonical projection
rebuilds complete run state from position zero alongside the conversation
records on which it depends.

This foundation is production-unused:

- `main.py`, `bot.py`, the conversation-run service, and all client APIs do not
  construct or call `WorkshopRunLifecycle` or
  `WorkshopRunExecutionAuthority`;
- it starts no process or worker and owns no lease, attempt, backend session,
  workspace, credential, or delivery;
- it does not alter Telegram ingress, streaming, outbox finalization, JSONL,
  memory, media, commands, schedules, integrations, or any backend driver;
- schema version 15 created an empty `runs` table; version 16 adds empty attempt
  state and nullable authority columns. Existing conversations replay without
  synthesizing historical runs or attempts.

The explicit run-authority cutover review is complete in section 27. It holds
live activation and proves that attempt identity is required for truthful
dispatch, crash recovery, and cancellation.

## 27. First run-authority cutover review

**Review date:** 2026-08-12

**Scope:** Whether the production-unused durable run lifecycle is sufficient
to emit live facts around the existing authenticated private-text conversation
service and trusted-host backend processes.

**Decision:** **Hold live lifecycle activation while adding a
production-unused durable execution-attempt and cancellation-intent
foundation.** The lifecycle vocabulary and projection are sound as durable
facts, but the current handler cannot place those facts around a local coding
agent without creating states that overclaim what Kai knows after a crash.

The blocker is not transport routing or Telegram final delivery. The durable
Telegram update queue and exact-epoch outbox already provide useful recovery
boundaries. The blocker is the non-transactional boundary between SQLite and a
backend process that may edit files, invoke tools, or affect remote systems
before Kai observes its first stream event. No database transaction can make
that external execution exactly once.

### 27.1 Current authority trace

For the live private-text path, authority currently crosses these boundaries:

1. webhook mode persists the raw Telegram update before acknowledging it;
2. the handler records the canonical inbound message idempotently;
3. the conversation service resolves one canonical human, channel, and agent,
   then privately resolves the current compatibility runtime key and model;
4. the per-conversation lock serializes dispatch;
5. `PreparedConversationRun.stream()` invokes the selected backend through the
   trusted-host `SubprocessPool`;
6. stable prefixes may create and edit a confirmed non-final Telegram preview;
7. a successful terminal `StreamEvent` yields one in-memory `AgentResponse`;
8. one transaction records the canonical assistant message, exact-epoch
   delivery request, and immutable edit/send operation plan;
9. the supervised outbox worker performs final Telegram delivery; and
10. only after the handler returns does the compatibility Telegram queue mark
    its update complete.

Polling mode enters the same handler without the compatibility webhook queue.
The canonical inbound message remains the transport-neutral idempotency fact
once handler execution begins.

### 27.2 Crash and replay findings

| Boundary | Durable evidence after a crash | Safe automatic action |
|---|---|---|
| Before canonical inbound commit | No Workshop command was accepted | Let the ingress transport retry according to its own contract |
| Inbound committed, run not accepted | Canonical message exists without a run | Deterministically repair acceptance; do not dispatch until acceptance commits |
| Run accepted, no execution authority granted | Accepted run has no attempt that may have executed | A worker may grant a new attempt once, under the channel/agent serialization policy |
| Attempt authority committed, before or during backend invocation | The backend may already have produced irreversible local or remote effects | Never automatically invoke the coding agent again; expire the owner and classify the attempt as interrupted for reconciliation |
| Backend success observed, terminal result not committed | Effects and an in-memory response may have existed, but no durable result exists | Do not regenerate automatically; classify interruption unless the same live owner can still commit the captured result |
| Canonical result, run completion, and delivery request committed | Durable terminal result and delivery authority exist | Suppress backend replay; let the outbox finish or resume delivery |
| Delivery occurred, ingress receipt not completed | Terminal run and binding-aware delivery evidence exist | Suppress backend replay and acknowledge the duplicate/replayed ingress receipt |

A lease can prove whether an owner is still entitled to act. Its expiry cannot
prove that the backend never ran. For coding agents, at-least-once redispatch
after the execution boundary is unsafe. The initial compatibility policy must
therefore be **retry before dispatch, never automatically retry after dispatch**.
A user may deliberately create a new run after inspecting an interrupted one.

### 27.3 Required semantic corrections before activation

The production-unused lifecycle can evolve safely before any historical live
run facts exist. Activation requires these corrections:

- **Atomic acceptance:** the authoritative private-text command must commit its
  canonical inbound message and `run.accepted` fact as one idempotent command
  transaction. Existing shadow adapters remain for excluded routes.
- **Execution attempts:** every dispatch needs a typed attempt identity, one
  current execution owner, a bounded lease, and a monotonic attempt sequence.
  The attempt snapshots backend-neutral execution selection needed for audit
  (agent, registered backend, provider when applicable, model, and execution
  contract), but never credentials, executable paths, prompts, or output.
- **Meaning of started:** `run.started` means Kai granted an attempt authority
  at the may-have-executed boundary. It does not claim that the first token was
  observed or that external effects are reversible.
- **Durable cancellation intent:** a human cancellation request is a separate
  fact from terminal cancellation. `run.cancelled` may be recorded only after
  the live execution owner confirms shutdown. The human is the request actor;
  the attached agent/execution authority is the terminal acknowledgement
  actor. The current human-authored terminal cancellation contract must change
  before activation. A crash with uncertain process outcome becomes bounded
  `execution_interrupted` failure evidence rather than a false cancellation
  acknowledgement.
- **Atomic successful terminal state:** `run.completed` must identify the
  canonical result message and commit in the same transaction as that message,
  its exact-epoch delivery request, and immutable operation plan. A completed
  run may never lack its durable visible result.
- **Durable failure/cancellation outcome:** terminal failure or cancellation
  and its bounded user-visible canonical outcome must commit before ingress is
  acknowledged. Native backend errors remain outside canonical facts.
- **Terminal replay suppression:** a replayed ingress receipt for a completed,
  failed, or cancelled run must not call the backend. Accepted work may proceed
  only through attempt authority; interrupted started work requires explicit
  human reconciliation or a new run.
- **Single terminal arbiter:** completion, failure, and confirmed cancellation
  race through one transactional state transition so exactly one terminal fact
  wins. A late `/stop` cannot rewrite a completed result.

### 27.4 Cutover gates

| Gate | Evidence | Result |
|---|---|---|
| Canonical target authorization | One human member and one attached agent resolve from the inbound `MessageId`; caller cannot select backend identity or pool key | Pass, production-used |
| Durable lifecycle vocabulary | Deterministic acceptance and terminal facts replay to an exact run projection | Pass, production-unused |
| Stable failure classification | Backend-native errors map conservatively to bounded backend-neutral categories | Pass as a building block |
| Durable final delivery | Canonical result and exact-epoch streaming-finalization plan recover without duplicate final delivery | Pass, production-used |
| Atomic inbound acceptance | Inbound message and run acceptance currently use separate service calls | Blocker |
| Attempt authority and ownership | Typed attempts, exclusive active ownership, leases, monotonic fencing, and conservative expiry recovery are replayed and tested | Pass, production-unused |
| Cancellation intent and acknowledgement | Human request and fenced acknowledgement are separate facts with one terminal arbiter; live `/stop` is not integrated | Pass as a production-unused foundation; integration blocker remains |
| Terminal result atomicity | Completion must reference a canonical result, but result creation, delivery request, and completion are not yet one transaction | Blocker |
| Terminal replay suppression | Replayed ingress has no run-state guard before backend invocation | Blocker |
| Installed qualification | No live run facts should exist until all preceding blockers pass in automated review | Not yet applicable |

### 27.5 Next bounded implementation milestone

Add a production-unused durable **run execution-authority foundation**. It
must:

1. introduce typed attempt identity and a replayed attempt projection linked to
   exactly one run;
2. grant at most one active execution owner per run, with lease and monotonic
   fencing semantics that prevent a stale owner from committing terminal state;
3. distinguish accepted/pre-dispatch work from may-have-executed work and
   forbid automatic redispatch of the latter after owner loss;
4. record human cancellation intent separately from terminal cancellation and
   define one transactional terminal-state arbiter;
5. require successful completion to reference one canonical result `MessageId`
   while leaving the later finalization integration production-unused;
6. retain only registered backend/provider/model identifiers and bounded status
   codes—never secrets, executable paths, raw errors, prompts, or output;
7. provide deterministic idempotency, replay, lease-expiry, stale-owner,
   completion/cancellation-race, and migration tests;
8. remain unregistered in `main.py`, `bot.py`, `sessions.py`, the conversation
   service, and every client endpoint.

Do not add a generic background execution worker or wire Telegram in this
slice. After the foundation passes, conduct a second run-authority review of
the atomic inbound/acceptance adapter and terminal-result finalization
integration before authorizing any live lifecycle facts.

## 28. Production-unused fenced run execution authority

**Implementation date:** 2026-08-12

Schema version 16 and canonical projection version 6 implement the bounded
foundation required by section 27 without registering a worker or altering any
live conversation path.

Each accepted run may have at most one active `run_attempt`. An attempt records
a typed attempt ID, monotonically increasing attempt sequence and fence token,
one opaque execution-owner ID, a bounded renewable lease, and the protected
backend/provider/model selection resolved from installation policy. It records
the execution contract identifier but never a command, executable path,
credential, prompt, output, or native backend error. Callers cannot supply the
selection, and a resolved backend must exist in the protected registry set.

The authority boundary is conservative:

- a lease grant is pre-dispatch and may expire back to an accepted run;
- `run.started` version 2 is emitted atomically with `run_attempt.started` at
  the may-have-executed boundary;
- an expired pre-dispatch grant may receive a new, higher fence;
- an expired started attempt becomes `execution_interrupted`, atomically fails
  its run, and cannot be automatically redispatched;
- renewals advance a lease version, so an older claim cannot start or settle;
- completion, failure, and confirmed cancellation require the exact active
  owner, fence, and lease version and race through one SQLite transaction;
- successful completion identifies one canonical agent result replying to the
  run's inbound message;
- human cancellation intent leaves the run nonterminal until the fenced owner
  confirms that execution stopped.

Deterministic event identities make exact retries return prior authority facts
while conflicting retries fail closed. Projection rules independently verify
actor, run, attempt, selection shape, lease ordering, cancellation intent,
result-message authorship, and terminal ordering. Rebuild tests restore the
same run and attempt state from position zero. Migration, stale-owner,
renewal, pre/post-dispatch expiry, protected selection, canonical result,
terminal conflict, and cancellation/completion race tests pin the contract.

The earlier lifecycle service now owns acceptance only. Its unfenced direct
start/complete/fail/cancel methods were removed so future wiring cannot choose
between competing transition authorities. Version-1 events remain replayable
for schema compatibility, although no production path emitted them.

This remains production-unused. No production module imports or constructs the
execution authority; it starts no process, owns no worker, and changes no
Telegram, backend, memory, media, command, schedule, integration, history, or
delivery behavior.

### 28.1 Next bounded milestone

Conduct the second run-authority integration review promised by section 27.
Specify the atomic command boundary for canonical inbound plus acceptance and
the atomic terminal boundary for canonical result or bounded failure outcome,
exact-epoch delivery work, and run settlement. The review must also define
terminal replay suppression and how the existing per-conversation lock maps to
one active attempt. Do not activate live run facts or add a generic execution
worker until that review closes every remaining blocker.

## 29. Second run-authority integration review

**Review date:** 2026-08-12

**Scope:** Whether the fenced authority in section 28 can now wrap the live
authenticated private-text path without duplicate coding-agent execution,
false terminal facts, or a transport-specific execution identity.

**Decision:** **Hold live activation. Implement the command and terminal
transaction coordinators as production-unused services before changing the
Telegram handler.** The attempt authority correctly fences an owner once a
run exists, but the surrounding live path still has four gaps across which a
retried Telegram update can invoke a backend without a durable proof that it
must not do so:

1. canonical inbound recording and run acceptance are separate commits;
2. the compatibility run snapshots a model separately from the protected
   backend instance that later executes;
3. canonical reply finalization and run settlement are separate transactions;
4. ingress replay does not inspect terminal or active-attempt state before
   calling the backend.

The current production path remains unchanged by this review. No run or
attempt facts are emitted, no worker is registered, and Telegram continues to
use the already-qualified conversation delivery authority.

### 29.1 Observed live boundary

The authenticated private-text handler currently performs these operations:

1. write the human message to transitional JSONL history;
2. append and project the canonical inbound Workshop message;
3. resolve the canonical human, channel, attached agent, private compatibility
   pool key, and a synchronous model value;
4. wait for the Telegram-chat lock;
5. call the compatibility pool, which may create the backend instance, restore
   workspace and persisted settings, and restart the process before sending;
6. publish optional, confirmed non-final Telegram streaming previews;
7. on success, commit the canonical assistant message, exact-epoch delivery
   request, and immutable edit/send operation plan in one transaction; and
8. return normally, after which webhook mode marks the durable Telegram update
   complete.

The final-delivery transaction is a strong boundary, but it contains no run or
attempt settlement. Backend failure, no response, `/stop`, preview-binding
failure, and finalization failure currently produce transitional JSONL and/or
direct Telegram notices rather than one canonical terminal outcome. Most of
those branches return normally, so webhook mode considers the ingress update
handled.

An uncaught exception or process exit instead causes the durable Telegram
queue to retry the same update. Because the handler has no run-state guard,
that retry can call the coding agent again even if the first process already
edited files, invoked tools, or changed a remote system. Polling enters the
same handler without the compatibility update queue, so duplicate-execution
suppression must belong to the canonical run boundary rather than to webhook
queue bookkeeping.

### 29.2 Atomic command acceptance

The authoritative command boundary must accept an authenticated client
message through one `BEGIN IMMEDIATE` transaction that:

1. resolves the external identity and channel binding;
2. appends the deterministic human `message.created` fact;
3. projects that message sufficiently to resolve its human membership and
   exactly one attached agent;
4. appends the deterministic `run.accepted` fact; and
5. projects and returns the resulting run plus a replay disposition.

The service must use transaction-local forms of inbound recording and run
acceptance. Calling the current public helpers in sequence would nest or split
transactions and preserve the crash gap. Exact retries must observe both prior
facts and verify that they describe identical content and authority. Finding
only one fact, an ambiguous binding, a changed body under the same transport
identity, a non-human author, or an ambiguous agent attachment must fail
closed before backend preparation.

Transitional JSONL history is not an authority input and need not share this
SQLite transaction. It may be updated after canonical acceptance, but its
failure or duplication may never decide whether the backend runs.

### 29.3 Protected execution preparation

`PreparedConversationRun.model` is not an execution-authority snapshot. The
current synchronous `SubprocessPool.get_model()` returns the global default
when no instance exists, while `send()` can later create a per-user backend
and apply persisted model settings. Even an existing instance can still have
pending settings restoration. Recording the former while executing the latter
would make the durable attempt audit false.

The protected preparation boundary must therefore resolve, under the
canonical execution lane, one hidden compatibility runtime and its effective
registered backend, provider, model, workspace, and process configuration
before the attempt grant. It must apply pending persisted settings before
snapshotting the selection. The same prepared runtime—not a second lookup—must
perform the fenced dispatch. The public command supplies only its canonical
run identity; it cannot supply a pool key, backend, provider, model, command,
executable, environment, or credential.

If the prepared runtime no longer matches the durable selection immediately
before dispatch, Kai must abandon the pre-dispatch grant and allow expiry
recovery before preparing a new higher-fenced attempt. It must not silently
change the selection attached to an active attempt.

### 29.4 Execution lane and dispatch point

The in-process serialization identity for an authoritative run is
`(channel_id, agent_id)`, derived from the accepted run. A Telegram chat ID is
only a compatibility adapter key and cannot remain the lock identity once a
desktop or another authenticated client can address the same channel.

For the first cutover, eligible private text must use one canonical lane
coordinator as its sole dispatch lock. Routes excluded from the cutover may
retain their current Telegram-chat locks. Taking both independently would
create two apparent locks for the same backend and would not provide a real
serialization guarantee.

Inside the canonical lane, dispatch proceeds in this order:

1. inspect the accepted run and any active or terminal attempt;
2. prepare the protected compatibility runtime and effective selection;
3. grant one pre-dispatch attempt with a bounded lease;
4. append `run_attempt.started` and `run.started` atomically immediately
   before calling the prepared runtime; and
5. invoke the backend only while the exact owner, fence, lease version, and
   prepared selection remain current.

The `started` commit is deliberately conservative. A crash after it commits
but before the process call is indistinguishable from a crash just after the
call. Both are treated as may-have-executed and are never automatically
redispatched. A grant that expires before `started` remains safe to replace
with a higher fence.

The database's unique active-attempt invariant is the durable authority; the
in-memory lane prevents ordinary local concurrency but is not relied upon for
restart safety.

### 29.5 Atomic terminal finalization

Successful completion requires one transaction that verifies the exact active
claim and atomically appends and projects:

- the deterministic canonical assistant result replying to the run's inbound
  message;
- the exact active-epoch delivery request;
- the immutable Telegram streaming-finalization operation plan;
- `run_attempt.completed`; and
- `run.completed`, referencing that canonical result `MessageId`.

No transaction may commit a completed run without its visible canonical
result and delivery work, or commit a successful result while leaving the run
started. The current outbound finalizer and execution authority both own their
transactions; integration therefore requires transaction-local primitives
behind one coordinator, not one public service calling the other.

Failure, confirmed cancellation, no response, and interrupted execution need
the same shape: a bounded, backend-neutral canonical user-visible outcome,
its exact-epoch delivery request and immutable operation plan, and the winning
attempt/run terminal facts must commit together. Native errors, prompts,
credentials, executable paths, and provider payloads remain excluded. A
confirmed non-final preview may be retained and finalized by the immutable
plan, but seeing a preview never authorizes backend replay.

Commit-uncertain handling must retry the entire deterministic transaction and
then inspect all expected facts. It must never fall back to direct Telegram
delivery or a second backend invocation when the first commit may have
succeeded.

### 29.6 Replay disposition

Every accepted command must receive one durable disposition before any
backend call:

| Durable state | Handler action |
|---|---|
| No canonical message/run | Atomically accept the command; do not dispatch before commit |
| Accepted, no active attempt | Eligible for protected preparation and one grant |
| Active pre-dispatch grant | Do not create another owner; the live owner continues or expiry returns the run to accepted |
| Started attempt | Never redispatch; the live owner may settle while its fence is current |
| Expired started attempt | Atomically record interrupted failure plus its visible delivery outcome; require explicit human retry as a new run |
| Completed, failed, or cancelled | Never call the backend; verify/resume the already-committed outbox work and finish the ingress receipt |

Webhook queue completion may occur once the terminal outcome and delivery
request are durably committed; it need not wait for Telegram delivery. A
nonterminal replay owned elsewhere must remain deferred or retryable rather
than returning success and losing the command. These rules also apply to
future clients even when they do not use the Telegram update queue.

The existing `recover_expired()` contract is sufficient for pre-dispatch
grant expiry, but its started-attempt path currently fails the run without a
canonical visible outcome or delivery request. That recovery path cannot be
registered in production until it uses the terminal coordinator.

### 29.7 Cancellation integration

`/stop` currently sets a Telegram-chat event, kills the compatibility pool
entry, and immediately tells Telegram that stopping began. It neither records
durable human intent nor identifies a fenced run owner.

The authoritative form must resolve the active run in the canonical
channel/agent lane, commit `run.cancellation_requested`, and then signal the
live owner associated with the exact claim. Only that owner may confirm
`run.cancelled`, and only after backend shutdown is known to have completed.
If Kai loses the owner or cannot prove shutdown, lease recovery records
`execution_interrupted` rather than falsely claiming cancellation. A late stop
against a terminal run is an idempotent no-op and cannot replace completion.

### 29.8 Cutover gates

| Gate | Current evidence | Result |
|---|---|---|
| Canonical target authorization | Message resolves one human member, channel, and attached agent | Pass, production-used |
| Fenced attempt authority | One active owner, monotonic fence, lease version, protected registry check, and conservative expiry | Pass, production-unused |
| Atomic command acceptance | Inbound recording and acceptance own separate transactions | Blocker |
| Truthful execution selection | Synchronous reported model can differ from the runtime selected by `send()` | Blocker |
| Canonical execution lane | Live lock and stop event are keyed by Telegram chat ID | Blocker |
| Started boundary | Authority contract is conservative and tested; live backend call is not wrapped | Integration blocker |
| Atomic successful terminal state | Reply/delivery/plan commit does not include attempt/run completion | Blocker |
| Atomic failure and cancellation outcome | Live branches use direct Telegram/JSONL; expired-start recovery has no visible outbox outcome | Blocker |
| Terminal replay suppression | Handler never checks run or attempt state before dispatch | Blocker |
| Generic execution worker | Not present and not yet justified | Correctly deferred |

### 29.9 Next bounded implementation sequence

The first implementation after this review is a production-unused **atomic
conversation-command acceptance service**. It must append canonical inbound
and `run.accepted` facts in one transaction, return a typed replay disposition,
and include deterministic conflict, rollback, projection-rebuild, and
duplicate-command tests. Existing production adapters remain unchanged.

Then, in separate bounded changes:

1. add protected asynchronous execution preparation that binds the effective
   registry selection to the exact compatibility runtime;
2. add a production-unused transaction coordinator for successful and bounded
   unsuccessful terminal outcomes, exact-epoch delivery work, and fenced run
   settlement;
3. add canonical lane and cancellation/recovery coordination without exposing
   Telegram IDs as execution authority;
4. exercise duplicate-ingress, crash-before-dispatch, crash-after-start,
   commit-uncertain, late-stop, and terminal-delivery recovery tests; and
5. conduct a final activation review before wiring authenticated private text.

Do not register a generic execution worker, emit live run facts, or change the
installed Telegram path in the command-acceptance slice.

## 30. Production-unused atomic conversation-command acceptance

**Implementation date:** 2026-08-12

The first bounded implementation from section 29 now provides one
production-unused `WorkshopConversationCommandService`. It accepts a validated
`InboundMessage` and owns a single `BEGIN IMMEDIATE` transaction spanning both
the deterministic canonical `message.created` event and its deterministic
`run.accepted` event. Both projections advance before commit, so no successful
call can expose a canonical command without its run or an accepted run without
its command.

The existing inbound recorder and run lifecycle retain their public contracts.
They now also expose transaction-local primitives that reject use without an
active caller-owned transaction. The command service is the only coordinator
of those primitives. If one event existed before the transaction and the other
would be new, the service rolls back and reports a state conflict instead of
silently repairing a half-authoritative command.

Exact retries verify the stored message content and acceptance authority, then
return a typed durable disposition:

- `newly_accepted`: both facts committed by this call;
- `ready_replay`: both facts already existed and no active attempt or
  cancellation request blocks later preparation;
- `active_replay`: a granted or started attempt already owns the run;
- `cancellation_pending_replay`: durable human cancellation intent exists;
- `terminal_replay`: the run is completed, failed, or cancelled.

No disposition invokes a backend. A started run without an active attempt,
multiple active attempts, changed content under the same transport identity,
ambiguous canonical attachment, or any partial prior state fails closed.
Concurrent duplicate commands on independent SQLite connections serialize to
one new acceptance and one ready replay. Rollback, exact retry, conflicting
content, partial-state rejection, active/cancellation/terminal classification,
and projection-rebuild tests pin the contract.

This foundation remains production-unused. `main.py`, `bot.py`, and
`sessions.py` do not import or construct it. It changes no Telegram ingress,
compatibility queue, backend process, lock, stop event, streaming preview,
outbox, JSONL history, memory, media, command, schedule, integration, or
installed behavior. No schema migration or generic worker was added.

### 30.1 Next bounded milestone

Add production-unused **protected asynchronous execution preparation**. It
must resolve the hidden compatibility runtime and fully effective registered
backend/provider/model selection as one object after pending persisted settings
have been applied. The exact prepared runtime must later perform dispatch, so
the attempt selection cannot differ from what executes. Callers continue to
supply only canonical run identity and never a transport pool key, backend,
provider, model, command, executable path, environment, or credential.

Do not grant live attempts, call a backend, change the Telegram handler, or
register a worker in that slice.

## 31. Protected execution preparation

**Implementation date:** 2026-08-12

The production-unused preparation service now resolves an accepted canonical
run to its hidden compatibility identity, applies pending workspace and user
settings, and snapshots the effective registered backend, provider, model, and
workspace from the exact runtime object. A one-shot prepared handle dispatches
only if that same pool object and its generic runtime fingerprint remain
current; replacement, pending settings, or selection/workspace/timeout drift
fails before the backend call. The transport pool key remains private.

The ordinary pool send path uses the same preparation primitive, preserving
existing behavior while eliminating the earlier difference between reported
and executed selection. Tests cover per-user selection overriding a different
global default, persisted setting application, exact-runtime dispatch,
one-shot use, drift rejection, registry enforcement, and cancellation intent.
No production module constructs the Workshop preparation service, so this
change emits no live attempt and performs no live dispatch.

### 31.1 Next bounded milestone

Add one production-unused terminal transaction coordinator that atomically
settles the fenced attempt/run together with the canonical result or bounded
failure outcome, exact-epoch delivery request, and immutable delivery plan.
Do not wire Telegram or register recovery until that coordinator and its
commit-uncertain tests pass.

## 32. Atomic terminal transaction coordinator

**Implementation date:** 2026-08-12

The production-unused terminal coordinator now commits one canonical visible
outcome, its exact active-epoch delivery request, immutable edit/send plan,
fenced attempt terminal fact, and run terminal fact in one `BEGIN IMMEDIATE`
transaction. Successful completion records the canonical result `MessageId`.
Failure accepts only typed backend-neutral codes and selects fixed bounded text
internally; confirmed cancellation resolves its durable human-requested code
and fixed visible text internally. Native backend errors cannot enter these
facts.

The existing outbound finalizer and execution authority now expose guarded
transaction-local primitives while their public behavior remains unchanged.
Exact retries require the message, delivery, plan, attempt, and run transition
to share one prior state. A stale fence, plan failure, partial prior state, or
losing terminal outcome rolls back without settling the run. If SQLite loses a
commit result, the coordinator repeats the whole deterministic transaction;
one unresolved retry becomes an explicit commit-uncertain error and never a
direct-send or backend-replay instruction.

Tests pin successful, failed, and cancelled settlement; exact replay; event
ordering; stale-fence and injected-plan rollback; partial-state rejection;
single-terminal arbitration; commit-result recovery; unresolved commit
uncertainty; and absence from production construction. No worker, Telegram
handler, recovery path, backend dispatch, schema migration, or installed
behavior changed.

### 32.1 Next bounded milestone

Add production-unused canonical lane coordination around preparation, attempt
grant, the conservative `started` boundary, and exact-runtime dispatch. Then
integrate durable cancellation and interrupted-execution recovery with that
lane. The lane identity must be canonical `(channel_id, agent_id)` and must not
expose a Telegram chat ID as execution authority.

## 33. Canonical execution coordinator

**Implementation date:** 2026-08-12

The production-unused coordinator now owns one canonical
`(channel_id, agent_id)` lane from an accepted `RunId` through protected
preparation, internal owner/fence grant, exact-runtime validation, the durable
started boundary, dispatch, lease renewal, and atomic terminal settlement. It
loads the prompt from the canonical inbound message; callers cannot supply a
transport identity, prompt override, backend, provider, model, command, path,
environment, or credential. Concurrent replay cannot invoke a second backend.

Cancellation records durable human intent, stops only the exact prepared
runtime, and confirms cancellation only after shutdown completes. A failed
shutdown becomes a bounded interruption rather than a false cancellation.
Pre-dispatch drift leaves an expirable grant and never calls the backend.
Expired started work now commits a fixed visible interruption, exact-epoch
delivery work, interrupted attempt, and failed run atomically; the obsolete
invisible-recovery path no longer exists. Native backend errors never enter
canonical history.

Focused tests cover success ordering, stored-prompt authority, duplicate
dispatch suppression, bounded failures, exact-runtime cancellation, drift and
grant recovery, post-start interruption, terminal replay, and absence from
production construction. No installed behavior changes.

### 33.1 Next bounded milestone

Conduct the activation review, then route authenticated private Telegram text
through this coordinator while retaining the current streaming presentation
and delivery worker. That activation is the first slice in this sequence that
will require `make install` and an operator-visible Telegram qualification.
