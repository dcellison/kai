# Kai Workshop Telegram Independence Plan

**Status:** Proposed migration plan  
**Date:** 2026-08-13  
**Scope:** Remove Telegram as a required host, identity source, and internal
routing authority while retaining Telegram as a fully supported Kai client and
delivery adapter.

## 1. Executive summary

Kai Workshop already has transport-independent principals, channels, agents,
messages, runs, attempts, client sessions, browser access, and durable delivery
records. The browser can enroll independently, submit commands, observe live
conversation history and run progress, cancel work, and receive canonical
results. Telegram is therefore no longer the only usable client.

Kai is not yet independent of Telegram internally. Startup still requires a
Telegram token, the Python service is assembled around a
`python-telegram-bot` `Application`, protected runtime profiles are derived from
Telegram-keyed `users.yaml` entries, and several compatibility stores and
services still use a Telegram-derived integer as their internal key. Scheduled
work, media handling, internal agent APIs, and some integrations also retain
Telegram-owned orchestration or delivery.

The target is not to remove Telegram. The target is this operational rule:

> Kai Workshop and its agents must start, run, recover, and remain fully usable
> when no Telegram adapter is configured. Enabling Telegram must add a client
> and delivery transport without changing core identity, execution, storage, or
> scheduling authority.

The highest-leverage work is therefore not a handler-by-handler rewrite. It is
to separate protected runtime policy from Telegram human configuration, create
a transport-neutral application host, and prove an installed Workshop-only
system. The remaining state and feature migrations can then proceed behind
stable service boundaries rather than creating more compatibility paths.

This document specializes the broader `kai-workshop-implementation-map.md`.
That map remains the architectural history and transition-retirement ledger;
this plan is the execution checklist for the specific Telegram-independence
boundary. Each cutover or compatibility mechanism introduced here must update
the map's retirement ledger with an explicit removal gate.

## 2. Meaning of “Telegram independence”

Telegram independence has four distinct requirements.

### 2.1 Startup independence

Kai can start without `TELEGRAM_BOT_TOKEN`, without constructing a Telegram
`Application`, and without contacting the Telegram API. Workshop HTTP routes,
client enrollment, canonical history, runtime execution, scheduling, internal
services, and applicable integrations remain available.

### 2.2 Identity independence

Canonical humans, agents, channels, runtime profiles, jobs, artifacts, and
memory scopes use opaque Workshop identifiers. Telegram user IDs, chat IDs,
message IDs, update IDs, and file IDs appear only in Telegram external-identity,
binding, ingress-deduplication, and delivery records.

### 2.3 Execution independence

Backend selection, OS execution identity, subprocess ownership, session
continuity, cancellation, workspace access, and run recovery are selected by
protected Workshop runtime policy. They do not depend on a Telegram user or
chat existing.

### 2.4 Feature independence

Core capabilities are services that any authorized client or trigger can use.
Telegram commands and media handlers translate Telegram input into those
services; they do not own the only implementation. A capability may still have
a Telegram-specific presentation, such as an inline keyboard or streaming
preview, without becoming Telegram-dependent.

## 3. What should remain Telegram-specific

The following are permanent adapter responsibilities and are not migration
debt:

- Telegram update parsing, webhook/polling ingress, and secret-token checks;
- Telegram sender authentication and external-identity resolution;
- mapping a Telegram user to a human principal;
- mapping a Telegram private chat or configured group to a canonical channel;
- Telegram update/message identity used for ingress idempotency;
- Telegram file download and transport metadata extraction;
- Telegram Markdown rendering, inline keyboards, chat actions, message-size
  limits, voice-note formatting, and streaming preview edits;
- Telegram delivery fragments, retries, ambiguity handling, and Bot API error
  classification;
- webhook registration, polling, Telegram health monitoring, and adapter
  diagnostics.

These responsibilities must live behind explicit ingress and delivery adapter
boundaries. They must not construct the core application or determine domain
identity.

## 4. Compatibility invariants

Every migration must preserve the following behavior:

1. Existing Telegram private conversations continue working.
2. GitHub notifications continue reaching the configured Telegram group.
3. Notification-only Telegram groups do not become inbound conversations.
4. Telegram streaming does not duplicate, reorder, or prematurely expose
   assistant output.
5. `/stop` remains responsive while an agent is executing.
6. TOTP settings and Telegram-sensitive command policy remain unchanged.
7. Existing uploads, history, memory, preferences, workspaces, and backend
   sessions remain readable through explicit migration adapters.
8. Claude, Codex, Goose, OpenCode, and Pi remain equal supported harnesses.
9. Protected backend commands continue to come only from the root-owned backend
   registry; client input cannot provide executable paths.
10. The configured OS execution user and workspace boundaries remain enforced.
11. Workshop browser enrollment, authorization, session revocation, timeline
    streaming, run progress, cancellation, and restart recovery remain intact.
12. `make config`, `make install`, and `make install-status` accurately describe
    deployed authority without exposing secrets.
13. No migration introduces a silent fallback from canonical authority to a
    Telegram-shaped legacy path.

## 5. Current transport-independent foundation

The following foundations are already in place and should be extended rather
than replaced:

- opaque `PrincipalId`, `ChannelId`, `AgentId`, `RuntimeProfileId`, `MessageId`,
  `RunId`, `AttemptId`, artifact, binding, delivery, device, and client-session
  identities;
- a canonical Workshop event store and deterministic projections;
- explicit workshop and channel memberships;
- transport bindings that distinguish external identities from canonical
  principals and channels;
- canonical human and agent provisioning;
- explicit channel-agent runtime-profile assignments;
- durable run acceptance, attempts, leases, fencing, terminal settlement,
  cancellation, reconciliation, and restart recovery;
- a profile-addressed Workshop runtime facade;
- canonical browser commands that do not require a Telegram identity or
  binding for authorization;
- a durable delivery outbox with binding-aware outcomes and Telegram workers;
- canonical notification feeds for ordinary GitHub notifications;
- a React client with enrollment, canonical timeline reads, resumable live
  events, command submission, activity reporting, cancellation, Markdown, and
  channel navigation;
- canonical active directories for files, history, managed homes, memory, and
  preferences.

This is enough foundation to remove Telegram hosting authority without
redesigning Workshop again.

## 6. Current coupling inventory

### 6.1 Protected runtime policy is derived from Telegram-keyed humans

`WorkshopRuntimeProfileRegistry.from_config()` currently iterates
`Config.user_configs`, requires each mapping key to equal `user.telegram_id`,
and derives the opaque runtime-profile ID from that integer. The protected
profile contains the correct backend, provider, OS user, and display name, but
its source identity is still a Telegram user record.

Relevant code:

- `src/kai/workshop/runtime_profiles.py`
- `src/kai/workshop/runtime_assignments.py`
- `src/kai/config.py`

Impact:

- a runtime profile cannot exist independently of a Telegram-shaped user entry;
- removing a Telegram identity risks appearing to remove execution policy;
- human identity and execution identity remain conceptually coupled;
- a future agent or shared channel cannot select a protected runtime without a
  compatibility configured-user integer behind it.

### 6.2 The subprocess pool still uses the compatibility integer

`WorkshopRuntimePool` accepts an opaque runtime profile, resolves its private
`runtime_config_id`, and calls the integer-keyed `SubprocessPool`. This facade is
a sound transition boundary, but the underlying pool still owns live backend
processes, in-flight state, restore state, settings, and workspace resolution
by the old key.

Relevant code:

- `src/kai/workshop/runtime_pool.py`
- `src/kai/pool.py`
- `src/kai/workshop/protected_execution.py`
- `src/kai/workshop/conversation_runs.py`

Impact:

- the core process lifecycle cannot yet operate solely on Workshop identities;
- one compatibility integer still joins runtime, settings, storage, and
  internal-agent credentials;
- removing the integer prematurely would break backend continuity and
  cancellation.

### 6.3 Kai’s application lifecycle is hosted by Telegram

`create_bot()` creates the Telegram `Application`, the `SubprocessPool`, the
Workshop runtime facade, and several Workshop services, then stores them in
`Application.bot_data`. `webhook.start()` requires that Telegram application,
extracts the pool and Workshop services from `bot_data`, and only then starts
the HTTP and Workshop client servers.

Relevant code:

- `src/kai/bot.py`
- `src/kai/webhook.py`
- `src/kai/main.py`

Impact:

- `TELEGRAM_BOT_TOKEN` is mandatory even for a browser-only installation;
- Workshop startup depends on successful Telegram application construction;
- lifecycle ownership is implicit and distributed between Telegram and HTTP
  modules;
- service dependencies are passed through a transport-owned dictionary rather
  than a typed core boundary;
- health can report the daemon as loaded without clearly distinguishing core,
  Workshop, runtime, scheduler, and adapter readiness.

### 6.4 Persistent compatibility state remains integer-keyed

Active filesystem directories have moved to canonical principal or channel
namespaces, but many database records and APIs still use `chat_id` or embed it
inside settings keys.

Current examples include:

- backend session continuity;
- model and timeout overrides;
- workspace selection and workspace-specific configuration;
- workspace history and allowed-workspace records;
- scheduled jobs and ownership checks;
- GitHub repository and notification overrides;
- locks and response-in-progress markers;
- internal agent credentials and allowed-service maps;
- parts of semantic-memory ingestion and retrieval.

Relevant code:

- `src/kai/sessions.py`
- `src/kai/locks.py`
- `src/kai/internal_api_auth.py`
- `src/kai/conversation_compatibility.py`
- `src/kai/workshop/compatibility_state.py`

The active directory changes in PRs #902–#905 are complete. Numeric directories
are now compatibility archives or read fallbacks, not the target write
authority. The records and callers behind those directories still require
migration.

### 6.5 Transcript context and memory still cross a compatibility adapter

Workshop browser execution writes canonical messages and runs, but
`WorkshopCompatibilityStateWriter` still appends JSONL history, saves the
backend session by compatibility integer, and schedules semantic-memory
ingestion with that integer. Backend context assembly can therefore still rely
on a legacy transcript namespace even though the browser renders canonical
messages.

Relevant code:

- `src/kai/workshop/compatibility_state.py`
- `src/kai/history.py`
- `src/kai/conversation_compatibility.py`
- `src/kai/memory.py`

Target:

- canonical messages are the sole conversational context source;
- backend session continuity belongs to a channel-agent runtime session;
- memory records carry explicit principal, channel, project, agent, and run
  provenance as appropriate;
- JSONL becomes an exporter/importer or archive format, not a production
  authority.

### 6.6 Non-text and command paths are not uniformly transport-neutral

Ordinary private text and Workshop browser commands cross canonical run and
delivery boundaries. Other Telegram routes still contain handler-owned domain
logic or direct delivery, including:

- photos and documents;
- voice transcription, voice modes, and voice output;
- file sending;
- model, settings, workspace, memory, job, GitHub, review, and related slash
  commands;
- some group behavior;
- specialized review and issue-triage paths.

Relevant code is concentrated in `src/kai/bot.py`, with related state in
`src/kai/sessions.py` and delivery in `src/kai/webhook.py`.

The correct target is not identical user interfaces. A Telegram inline keyboard
may remain unique to Telegram. The action behind it—such as changing a model or
selecting a workspace—must be an authorized service command that other clients
can invoke through their own presentation.

### 6.7 Scheduling is owned by the Telegram application

The scheduler uses `python-telegram-bot`’s APScheduler `JobQueue`, obtains the
pool and configuration through `bot_data`, executes backends directly, and
sends results through `context.bot`.

Relevant code:

- `src/kai/cron.py`
- scheduling routes in `src/kai/webhook.py`
- job persistence in `src/kai/sessions.py`

Target:

- Kai owns its scheduler lifecycle;
- a firing schedule creates the same durable Workshop run or notification
  command used by an interactive client;
- the job records canonical creator, channel, agent, runtime assignment, and
  optional project workspace;
- delivery is requested through configured channel bindings and the outbox;
- Telegram is only one possible delivery binding.

### 6.8 Internal agent APIs expose compatibility identity

Internal API credentials and service permissions are still organized by the
compatibility integer. Agents are instructed to supply or operate with
`chat_id` for scheduling, message, and file operations. Authentication already
prevents credentials from selecting arbitrary principals, but the public shape
still reflects Telegram-era identity.

Relevant code:

- `src/kai/internal_api_auth.py`
- internal routes in `src/kai/webhook.py`
- generated agent context and service instructions;
- sudoers and protected-agent credential construction.

Target:

- a backend credential resolves server-side to an execution principal,
  runtime profile, agent, run/attempt, and permitted scopes;
- callers do not submit a Telegram user or chat ID to select authority;
- message destinations are canonical channels or pre-authorized bindings;
- file operations use artifact IDs and scoped storage roots;
- schedule operations use canonical channel/agent/project authority;
- notification-only credentials remain send-only.

### 6.9 Integrations retain mixed authority

Ordinary GitHub notifications targeting a canonical notification channel now
appear in Workshop and use durable Telegram delivery. Compatibility routing
still exists for destinations without a canonical binding, and specialized
review, issue-triage, generic webhook, and scheduled integration paths do not
all enter the same canonical command/event boundary.

Relevant code:

- GitHub and generic webhook routes in `src/kai/webhook.py`;
- GitHub settings in `src/kai/sessions.py`;
- review and triage orchestration;
- `src/kai/workshop/telegram_delivery_runtime.py`.

Target:

- integrations authenticate and create typed canonical events or commands;
- routing resolves a canonical channel, agent, project, or notification feed;
- transport delivery is a separate outbox effect;
- external idempotency keys prevent duplicate events and deliveries;
- an unbound Telegram destination is an explicit configuration error or a
  documented supported adapter route, never a silent domain fallback.

## 7. Target architecture

The target process composition is:

```text
Protected operator policy
  |-- backend registry
  |-- runtime profiles
  |-- bootstrap/recovery policy
  `-- enabled adapters
             |
             v
       Kai application host
  +---------------------------------------------+
  | canonical store and projections             |
  | client identity/session service             |
  | command acceptance and authorization        |
  | run coordinator and executor                |
  | runtime-profile registry and runtime backend|
  | artifact and memory services                |
  | scheduler                                   |
  | integration services                        |
  | durable delivery outbox and supervisors     |
  | health/readiness                             |
  +---------------------------------------------+
       ^             ^             ^
       |             |             |
  Workshop HTTP   Telegram      GitHub/generic
  client adapter  adapter       ingress adapters
```

The Kai application host owns construction, startup order, readiness, task
supervision, graceful shutdown, and resource cleanup. No transport object owns
core services.

### 7.1 Suggested typed service boundary

The exact class names may change, but startup should construct a typed object
with responsibilities equivalent to:

```text
KaiApplicationServices
  config / protected policy
  database and Workshop event store
  runtime-profile registry
  RuntimeBackend / WorkshopRuntimePool
  command and authorization services
  run coordinator and executor
  canonical context/session service
  artifact and memory services
  scheduler
  integration registry
  delivery outbox and worker supervisor
  client enrollment/session service
  health and readiness service
```

Adapters receive only the services they require. Telegram does not receive
private database internals or unrestricted runtime selection merely because it
is enabled.

### 7.2 Supported deployment modes

The final host should support three explicit modes:

1. **Workshop-only:** browser/desktop clients, no Telegram token or connection.
2. **Hybrid:** Workshop plus Telegram, preserving the current installation’s
   behavior.
3. **Telegram-only client surface:** supported where desired, while still using
   the same canonical core and application host.

These are adapter configurations, not different domain architectures.

## 8. Protected runtime-profile design

### 8.1 Separate runtime profiles from humans

A runtime profile is protected execution policy, not a person. It should have
an operator-controlled stable identifier and contain or reference:

- OS execution user;
- backend registry key;
- provider where the harness requires one;
- model policy validated against that backend’s allowed model catalog, while
  the ordinary default remains the backend registry’s marked default;
- timeout and resource policy defaults;
- workspace or project access policy;
- internal service scopes;
- future worker/runtime isolation policy.

A Workshop channel-agent assignment refers to the runtime profile. A human may
own several channels, several agents may use different profiles, and a runtime
profile need not encode a Telegram identity.

### 8.2 Recommended configuration boundary

Use a protected runtime-policy document separate from transport identities.
The final filename is an implementation choice; conceptually it should resemble:

```yaml
version: 1
runtime_profiles:
  daniel-coding:
    os_user: daniel
    backend: codex
  scott-coding:
    os_user: sellison
    backend: claude
```

The profile’s `backend` must reference `/etc/kai/backends.yaml`; clients and
ordinary users never provide executable paths. Model validation uses the
backend registry, including its marked default. A profile-specific model
override may exist as explicit policy or mutable Workshop state, but runtime
profile creation must not require the operator to choose a default already
defined by the registry. No backend is assumed to exist and no backend receives
special fallback priority.

`users.yaml` may remain temporarily as protected Telegram/bootstrap
configuration, but its key must cease to be execution identity. Longer term,
mutable Workshop humans and assignments belong in canonical administration;
protected files retain only installation policy, recovery bootstrap, and
secrets.

### 8.3 Migration of existing installations

The installer must:

1. derive one stable runtime profile for each current configured execution
   identity without changing its backend, model, OS user, workspace, or scopes;
2. preserve existing `rtp_` IDs or append explicit reassignment events so
   projection replay remains deterministic;
3. validate every channel-agent assignment against protected runtime policy;
4. report planned mappings in dry-run/status output without exposing secrets;
5. refuse ambiguous or missing mappings rather than choosing a backend;
6. leave the existing configuration recoverable during a bounded rollback
   window;
7. stop deriving new runtime profiles from Telegram IDs after cutover.

## 9. Transport-neutral application lifecycle

### 9.1 Construction order

The core host should initialize in this order:

1. load and validate protected policy;
2. open the database and run additive migrations;
3. bootstrap/reconcile canonical Workshop records;
4. construct runtime profiles and runtime backend;
5. construct artifact, context, memory, command, run, scheduler, integration,
   and delivery services;
6. reconcile durable runs, deliveries, and schedules;
7. start supervised workers;
8. start Workshop client listeners;
9. start each configured external adapter;
10. declare readiness only when every required component is ready.

An optional adapter failure must be represented explicitly. Policy decides
whether it makes the whole service unavailable or leaves the core ready with a
degraded adapter.

### 9.2 Shutdown order

Shutdown should:

1. stop accepting new external input;
2. mark the service as draining;
3. stop adapter ingress;
4. stop schedule dispatch;
5. cancel or drain supervised tasks according to durable run policy;
6. settle or safely abandon delivery leases;
7. stop backend processes;
8. close HTTP listeners and database connections.

Telegram webhook deletion belongs to the Telegram adapter’s shutdown, not the
core host’s shutdown.

### 9.3 Configuration behavior

`TELEGRAM_BOT_TOKEN` is required only when the Telegram adapter is enabled.
Likewise, webhook URL and secret validation runs only for Telegram webhook
mode. `make config` should expose adapter enablement explicitly and preserve
existing hybrid installations without forcing re-entry of unchanged secrets.

## 10. Canonical state migration

State should migrate according to what owns it, rather than replacing every
`chat_id` column with the same opaque ID.

| Current state | Correct target owner |
|---|---|
| Backend provider session | channel + agent + runtime profile |
| Personal preferences | human principal |
| Channel behavior | channel |
| Agent model/runtime policy | channel-agent assignment or runtime profile |
| Active project/workspace | channel-agent session or project workspace |
| Allowed filesystem roots | protected runtime profile / project grant |
| Workspace history | principal and/or project workspace |
| Scheduled job | workshop + creator principal + target channel + agent |
| GitHub subscriptions | integration installation/subscription + target channel |
| Conversation lock | channel-agent execution lane or run |
| Upload ownership | artifact + principal/channel provenance |
| Memory | explicit principal/channel/project/agent/run scope |
| Internal API credential | runtime profile + agent + run/attempt + scopes |

Migration rules:

- add canonical columns/tables before changing reads;
- backfill deterministically through existing principal/channel/runtime
  mappings;
- dual-read only for a bounded, measured migration window;
- do not continue dual writes after canonical authority is proven;
- report unmapped and ambiguous rows in `make install-status`;
- archive or explicitly migrate numeric data; never silently discard it;
- delete obsolete keys and tables only in a separate retirement migration.

## 11. Canonical context, sessions, and memory

### 11.1 Context assembly

The agent prompt should be built from a snapshot-stable canonical channel
timeline, relevant thread/run state, selected project workspace, and authorized
memory. Telegram JSONL must not be necessary to reconstruct a conversation.

### 11.2 Backend session continuity

Provider session IDs belong to the channel-agent runtime session. The durable
record should include:

- channel and agent IDs;
- runtime profile ID;
- provider/backend session reference;
- effective model and workspace/project context;
- creation and last-use timestamps;
- invalidation/restart reason;
- optional lineage when a session is replaced.

### 11.3 Memory ingestion

Memory ingestion should consume canonical message and run IDs. It must preserve
speaker attribution and attach explicit scope/provenance. Compatibility
integers must not determine visibility.

### 11.4 JSONL retirement

After canonical context and memory qualification:

- stop production JSONL writes;
- remove JSONL reads from context, memory, and session recovery;
- retain a deliberate export command if human-readable transcripts remain
  valuable;
- retain an importer only if recovery from historical archives is supported;
- remove parity diagnostics once there are no two live authorities to compare.

## 12. Transport-neutral feature services

Telegram handlers should become thin adapters around typed services.

### 12.1 Settings and workspace actions

Create authorized service operations for model selection, timeout, workspace,
project, environment, prompt, voice preference, and memory controls. Each
operation receives authenticated canonical authority and validates policy on
the server. Telegram keyboards and Workshop controls invoke the same operation.

### 12.2 Media and artifacts

A client-neutral artifact-ingress service should:

- accept authenticated principal/channel provenance;
- validate size, media type, and filename independently of transport claims;
- store bytes under canonical artifact identity;
- record content hash and immutable provenance;
- optionally launch transcription or image/document processing;
- attach the result to a canonical message/run;
- expose only authorized artifact IDs to agents and clients.

Telegram file download remains in the Telegram adapter. Browser upload and a
future desktop client use their own byte transport into the same service.

### 12.3 Command presentation

Some slash commands may remain convenient Telegram syntax. They should map to
canonical commands rather than defining capabilities available nowhere else.
Transport-only commands such as Telegram help text may remain adapter-local.

## 13. Scheduler migration

Kai should own an `AsyncIOScheduler` or equivalent scheduler directly rather
than receiving it from `python-telegram-bot`.

A scheduled firing must atomically or idempotently create one of:

- a canonical reminder/notification message plus delivery requests; or
- a canonical run command targeting a channel-agent assignment.

Required scheduler properties:

- definitions and firing identity survive restart;
- duplicate firing cannot create duplicate runs or delivery;
- cancellation and rescheduling are authorized by canonical ownership;
- failed execution is visible in Workshop;
- reminders can reach any configured client binding;
- agent work uses the same run coordinator as browser and Telegram commands;
- no scheduler code calls `context.bot` or the subprocess pool directly.

## 14. Internal agent API migration

The internal API is a security boundary and should migrate after runtime
profiles are independent but before compatibility integers are retired.

### 14.1 Credential authority

Issue credentials for a concrete execution context. Server-side authentication
resolves:

- runtime profile;
- agent;
- current run and attempt where applicable;
- owning principal/channel/project grants;
- explicit scopes and allowed services.

### 14.2 Request shape

Remove caller-provided `chat_id` as an authority selector. Endpoints should
either act on the credential’s current context or accept a canonical resource
ID that the server authorizes against that context.

Examples:

- send a message to an authorized canonical channel;
- create a schedule for the current authorized channel/agent;
- read or write an authorized artifact;
- access a registered service by scope;
- request a delivery through a named authorized binding.

### 14.3 Compatibility and revocation

Existing credentials must have an explicit invalidation boundary. Startup must
not silently interpret an old numeric credential as a new canonical principal.
Tests must prove scope separation, revocation, cross-principal denial, and
send-only notification credentials.

## 15. Integration migration

Each integration follows the same pipeline:

```text
authenticated external event
  -> normalized integration event / command
  -> canonical authorization and routing
  -> canonical notification or run
  -> durable delivery requests for configured bindings
```

### 15.1 GitHub

Extend the canonical notification-feed path to specialized review and issue
triage. Repository subscriptions and overrides should belong to a principal,
workshop, or integration installation and route to a canonical channel.

### 15.2 Generic webhooks

Preserve the authenticated external contract while replacing the implicit
Telegram destination with an explicit canonical channel or integration route.
Reject ambiguous targets.

### 15.3 Delivery

No integration should call Telegram directly after its cutover. Telegram group
behavior is preserved by a channel binding and the delivery worker. Browser
visibility is produced by the canonical notification message, not by copying a
Telegram delivery result back into Workshop.

## 16. Implementation milestones

The work should be delivered as coherent milestones, not a separate PR for
every renamed field or helper.

### Milestone 1: Independent protected runtime policy

Deliverables:

- first-class runtime-profile configuration independent of Telegram users;
- deterministic migration of existing installations;
- profile validation against the backend registry;
- unchanged channel-agent assignments and five-backend behavior;
- installer dry-run/status diagnostics;
- no runtime profile derived from a newly supplied Telegram ID after cutover.

Exit gate:

- provision a non-Telegram human and assign an independently configured runtime
  profile without adding a fake Telegram user;
- existing Daniel and Scott profiles retain backend, OS user, model, workspace,
  session continuity, and isolation.

### Milestone 2: Transport-neutral application host

Deliverables:

- typed core service container;
- core-owned runtime, executor, outbox, scheduler, stores, client API, health,
  and shutdown lifecycle;
- Telegram adapter constructed only when enabled;
- removal of core service lookup through Telegram `bot_data`;
- component readiness and degraded-adapter reporting.

Exit gate:

- unit and integration tests construct the core without importing or mocking a
  Telegram `Application`;
- hybrid installed operation remains unchanged.

### Milestone 3: Workshop-only installed qualification

Deliverables:

- optional Telegram token and adapter configuration;
- an install/status path that accurately reports Telegram disabled;
- tests for Workshop-only, hybrid, and Telegram-client deployment modes;
- operator documentation for enabling or adding Telegram later.

Installed exit gate:

1. start Kai with no Telegram token;
2. enroll a Workshop browser;
3. submit a command to an assigned agent;
4. observe live run activity and final canonical output;
5. cancel a long run;
6. restart Kai during or between work and verify recovery;
7. verify schedules and applicable integrations do not require Telegram;
8. verify no process contacts Telegram;
9. restore hybrid configuration and confirm private chat and notification-group
   delivery.

### Milestone 4: Canonical state and context authority

Deliverables:

- canonical ownership for sessions, settings, workspaces, jobs, GitHub
  subscriptions, locks, and memory provenance;
- canonical timeline context assembly;
- channel-agent backend session records;
- bounded backfill and compatibility diagnostics;
- JSONL export/archive replacement.

Exit gate:

- a Workshop-only conversation survives restart with correct context, backend
  session, workspace, settings, and memory while all numeric compatibility
  stores are read-disabled in qualification mode.

### Milestone 5: Shared feature and integration services

Deliverables:

- transport-neutral media/artifact processing;
- authorized settings/workspace/memory operations;
- core-owned scheduling and scheduled run creation;
- canonical GitHub review/triage and generic webhook routing;
- outbox-only external delivery after each bounded cutover.

Exit gate:

- Telegram and Workshop exercise equivalent domain operations where each has a
  UI;
- scheduled and integration-triggered work is visible and recoverable as
  canonical runs/messages;
- configured Telegram delivery remains unchanged.

### Milestone 6: Scoped internal APIs and retirement

Deliverables:

- canonical execution credentials and request schemas;
- removal of caller-supplied compatibility identity;
- retirement of integer-keyed pool/state adapters;
- removal of numeric directory fallbacks after archive/migration approval;
- removal of shadow recorders, parity diagnostics, direct-send fallbacks, and
  obsolete schema;
- final documentation and installer simplification.

Exit gate:

- repository-wide searches find Telegram IDs only in adapter, external-binding,
  ingress-idempotency, delivery, migration/archive, and deliberately supported
  Telegram configuration code;
- disabling Telegram changes only adapter health and delivery availability;
- no core capability, identity, or runtime selection changes.

## 17. Testing and qualification matrix

### 17.1 Automated layers

- domain and schema tests for opaque identity and authorization;
- deterministic migration and replay tests;
- runtime-profile registry tests with no configured Telegram users;
- application-host lifecycle tests with zero adapters, Workshop only, Telegram
  only, and hybrid mode;
- scheduler restart, duplicate-fire, cancellation, and recovery tests;
- canonical context/session continuity tests;
- internal API scope and cross-principal denial tests;
- media/artifact provenance and authorization tests;
- integration idempotency and routing tests;
- outbox ordering, retry, restart, and ambiguity tests;
- static checks preventing core modules from importing Telegram packages where
  the architecture forbids it.

### 17.2 Five-backend matrix

Where a milestone touches execution, test Claude, Codex, Goose, OpenCode, and
Pi through the same runtime-profile contract. Backend-specific limitations must
be declared capabilities, not conditional core logic or fallback priority.

### 17.3 Installed hybrid qualification

At every production-affecting cutover:

- Kai starts and responds in Telegram;
- Workshop starts and responds in the browser;
- both clients see one canonical conversation result;
- long output, cancellation, and restart behavior remain correct;
- files and voice are tested when affected;
- GitHub notification-group delivery is tested when affected;
- `make install-status` reports clean authority and no unexplained migration
  backlog.

### 17.4 Installed Workshop-only qualification

Workshop-only qualification must use a configuration that contains no Telegram
token and no synthetic Telegram identity. It is not sufficient to disable
network access while retaining Telegram-shaped runtime records.

## 18. Observability and operator experience

`make install-status` should evolve to report, without secret values:

- core application readiness;
- enabled adapters and per-adapter readiness/degraded state;
- runtime-profile count and assignment coverage;
- Workshop principal/channel/agent/run/delivery health;
- scheduler readiness and nonterminal firing count;
- canonical state migration counts;
- unmapped or ambiguous compatibility records;
- legacy read/write authority status;
- whether JSONL and numeric-directory fallbacks remain enabled;
- version and configuration schema versions.

Error messages should identify the failed boundary. For example, “Telegram
adapter authentication expired” must not look like “Kai failed to start” when
Workshop and its agents are healthy.

## 19. Security requirements

- Runtime policy and adapter secrets remain root-owned protected configuration.
- Human clients cannot provide executable paths, OS users, runtime profiles,
  providers, models outside policy, or filesystem grants.
- No backend is assumed or used as a fallback.
- A missing runtime assignment fails closed.
- External identity never grants Workshop membership by itself.
- Telegram notification destinations never grant inbound authority.
- Client bearer sessions and backend internal credentials remain separate trust
  domains.
- Internal credentials are least-privilege and bound to server-resolved
  canonical execution context.
- Optional adapter disablement must not weaken authentication on remaining
  surfaces.
- LAN Workshop access retains its trusted-network/TLS boundary and restrictive
  browser security headers.
- State migration must preserve ownership and permissions and must never copy
  one principal’s data into another principal’s namespace.

## 20. Rollback and data safety

Each authority cutover requires:

- a database backup or documented restore point;
- explicit old-to-new record mapping;
- idempotent migration that can be rerun;
- a dry-run or diagnostic preview where practical;
- no destructive deletion in the cutover PR;
- a bounded rollback window with declared limitations;
- a later dedicated retirement change after installed evidence;
- explicit archive policy for numeric directories and JSONL.

Rollback is not permission to retain dual authority indefinitely. Once the new
path is proven and the rollback window closes, old writes and silent fallbacks
must be removed.

## 21. Work that should wait

The following valuable work should not interrupt Milestones 1–3 unless it fixes
a concrete correctness or security defect:

- `@agent` addressing in shared conversations;
- the detailed live agent tool/diff inspector;
- broad Workshop administration UI;
- runtime themes;
- desktop packaging;
- isolated/containerized workers;
- multi-workshop administration;
- large visual refinements.

Those features will be easier to implement once the application host and
runtime policy are genuinely transport-neutral. Containerized workers remain
the intended stronger execution boundary, but they should consume the runtime
profile and host contracts established here rather than block Telegram
independence.

## 22. Completion criteria

Telegram independence is complete only when all of the following are true:

- Kai installs and starts with Telegram disabled and no token present;
- Workshop-only humans and runtime profiles require no Telegram-shaped records;
- core services are constructed and supervised outside the Telegram adapter;
- browser/desktop clients can use agents, settings, workspaces, artifacts,
  schedules, memory, integrations, and recovery without Telegram;
- subprocesses and persistent state use canonical runtime/domain identities;
- canonical messages are the context authority;
- scheduled and integration work creates canonical runs/messages;
- internal agent APIs use scoped canonical execution credentials;
- Telegram sends occur only through the Telegram delivery adapter/worker;
- Telegram IDs occur only at legitimate Telegram adapter/binding boundaries;
- legacy numeric namespaces, JSONL authority, shadow writers, fallback routes,
  and transition-only diagnostics are retired or explicitly documented as
  supported archive/import facilities;
- enabling Telegram adds Telegram behavior without changing core execution;
- all five harnesses pass the common installed qualification contract.

At that point Telegram is exactly what Workshop intends it to be: one excellent
client and delivery transport among several, with no privileged position in
Kai’s backend architecture.
