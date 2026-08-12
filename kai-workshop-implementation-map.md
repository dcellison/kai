# Kai Workshop: Phase 0 Implementation Map

**Status:** Active migration plan; canonical conversation shadow implemented, production authority held
**Date:** 2026-08-12
**Scope:** Map the current Kai implementation onto the proposed Kai Workshop architecture without changing production behavior.

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
| Active harness process | `SubprocessPool` in [`pool.py`](src/kai/pool.py) | Chat ID | One implicit agent process per conversation key; no durable agent, run, attempt, lease, or worker identity. |
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

The delivery foundation now adds only the request and outcome facts the system can define consistently; mutable leases and attempts remain operational outbox state rather than replayed collaboration events. Run event families should be designed before execution becomes authoritative. The schema should not invent detailed tool or backend-worker events that the current harnesses cannot yet emit consistently.

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
| Authenticated plain-text/photo/document/voice and successful assistant-result canonical shadow writes beside current history, plus non-authoritative Telegram delivery observations | Workshop event store, message/artifact projections, and durable delivery outbox | Deterministic replay, duplicate-ingress/result tests, restart tests, full media-ingress coverage, delivery parity diagnostics, and sustained parity with current history and Telegram outcomes | Remove the `workshop_inbound_recorder`, `workshop_artifact_recorder`, `workshop_outbound_recorder`, and `workshop_delivery_recorder` bot-data adapters and their fail-open handler branches; remove the transitional `workshop_message_shadowed` JSONL marker; remove `workshop_message_parity_status` and its install-status output after canonical reads and outbox delivery become authoritative; make canonical command/event transactions and the delivery outbox the sole write and delivery paths | Active (inbound text/photo/document/voice, photo/document/voice artifacts, assistant results, delivery observations, read-only parity diagnostic, production-unused outbox and Telegram worker foundations) |
| JSONL transcript writes and reads | Canonical message projection, with an explicit export facility if still useful | Canonical reads serve complete context and transcript views; migration/parity diagnostics report no unexplained divergence | Stop JSONL writes; remove production reads and dual-write recovery code; retain only a documented importer/exporter if required | Planned |
| Telegram `chat_id` used as internal identity, namespace, and routing key | Durable principal, channel, agent, and binding IDs | Private chats, notification-only groups, duplicate updates, and restart routing all resolve correctly through bindings | Confine Telegram IDs to external identity, transport binding, and idempotency records; remove chat-shaped domain keys | Planned |
| `SubprocessPool` keyed by Telegram chat ID | Durable channel/agent session plus run and attempt orchestration | All five harnesses pass continuity, restart, cancellation, and isolation tests through durable identities | Remove chat-key compatibility lookup and move lifecycle ownership behind the orchestrator/runtime contract | Planned |
| Direct backend invocation from Telegram handlers | Transport-neutral command and run services | Telegram and the first Workshop client produce equivalent authorized runs and visible results | Remove handler-owned orchestration; leave authentication, parsing, and rendering in the Telegram adapter | Planned |
| Direct Telegram delivery from handlers, schedules, and webhooks | Durable delivery outbox and Telegram delivery adapter | Delivery outcome events preserve binding identity; retry, crash recovery, ordering, private-chat and notification-group delivery tests pass; live delivery is verified | Register the outbox worker only in an explicit cutover; remove direct Bot API sends from domain paths and delete delivery fallback flags after installed verification | Active (production-unused durable request/lease/attempt/retry/recovery, fragment progress, binding-aware terminal outcome facts, per-binding FIFO claims, atomic canonical assistant-result plus delivery-request transaction, canonical outbound-only notification channels, Telegram adapter/worker, and explicit runtime owner; installed direct-chat recovery and notification-group delivery passed, while production cutover is held on durable streaming finalization, work classification, and lifecycle integration) |
| Operator-invoked Workshop delivery qualification CLI | Installed evidence followed by the production delivery worker | A configured direct-chat reply is prepared without sending, survives a service restart, recovers an intentionally abandoned lease, reaches Telegram once through the exact selected delivery, and records a terminal binding-aware outcome; a configured notification group resolves through its outbound-only canonical channel, receives one atomically prepared qualification message through the exact selected delivery, and does not become an inbound conversation | Remove the qualification command and its explicit-claim-only surface after the production worker has equivalent installed restart/recovery evidence and direct delivery is retired | Active (the installed direct-chat recovery and notification-group delivery gates passed on 2026-08-12; retain until equivalent production-worker evidence exists, while the command remains unregistered and incapable of draining unrelated work) |
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
