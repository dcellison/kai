(() => {
  "use strict";

  const SESSION_KEY = "kai.workshop.read-session.v1";
  const CHANNEL_PATTERN = /^chn_[0-9a-f]{32}$/;
  const RECONNECT_DELAY_MS = 2000;

  const enrollmentPanel = document.querySelector("#enrollment-panel");
  const enrollmentForm = document.querySelector("#enrollment-form");
  const enrollmentError = document.querySelector("#enrollment-error");
  const credentialFields = document.querySelector("#credential-fields");
  const enrolledSessionHint = document.querySelector("#enrolled-session-hint");
  const deviceNameInput = document.querySelector("#device-name");
  const channelInput = document.querySelector("#channel-id");
  const enrollmentTokenInput = document.querySelector("#enrollment-token");
  const forgetEnrollmentSessionButton = document.querySelector(
    "#forget-enrollment-session",
  );
  const timelinePanel = document.querySelector("#timeline-panel");
  const timeline = document.querySelector("#timeline");
  const emptyTimeline = document.querySelector("#empty-timeline");
  const channelTitle = document.querySelector("#channel-title");
  const connectionStatus = document.querySelector("#connection-status");
  const connectionLabel = document.querySelector("#connection-label");
  const forgetSessionButton = document.querySelector("#forget-session");
  const messageTemplate = document.querySelector("#message-template");

  const state = {
    token: null,
    channelId: null,
    lastEventId: null,
    controller: null,
    generation: 0,
    seenMessageIds: new Set(),
  };

  class AuthenticationError extends Error {}
  class ChannelAccessError extends Error {}
  class ResynchronizationRequired extends Error {}

  function setStatus(label, kind = "connecting") {
    connectionLabel.textContent = label;
    connectionStatus.className = `status ${kind}`;
  }

  function showEnrollmentError(message) {
    enrollmentError.textContent = message;
    enrollmentError.hidden = false;
  }

  function setEnrollmentMode(hasSession) {
    credentialFields.hidden = hasSession;
    enrolledSessionHint.hidden = !hasSession;
    forgetEnrollmentSessionButton.hidden = !hasSession;
    deviceNameInput.required = !hasSession;
    enrollmentTokenInput.required = !hasSession;
  }

  function safeErrorMessage(payload, fallback) {
    const message = payload?.error?.message;
    return typeof message === "string" && message.length <= 200 ? message : fallback;
  }

  async function responsePayload(response) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  function saveSession() {
    sessionStorage.setItem(
      SESSION_KEY,
      JSON.stringify({ token: state.token, channelId: state.channelId }),
    );
  }

  function restoreSession() {
    try {
      const stored = JSON.parse(sessionStorage.getItem(SESSION_KEY));
      if (
        typeof stored?.token === "string" &&
        stored.token.length > 0 &&
        typeof stored?.channelId === "string" &&
        CHANNEL_PATTERN.test(stored.channelId)
      ) {
        state.token = stored.token;
        state.channelId = stored.channelId;
        return true;
      }
    } catch {
      // A malformed tab-local value has no authority and is discarded below.
    }
    sessionStorage.removeItem(SESSION_KEY);
    return false;
  }

  function stopStream() {
    state.generation += 1;
    state.controller?.abort();
    state.controller = null;
  }

  function forgetSession(message = null) {
    stopStream();
    state.token = null;
    state.channelId = null;
    state.lastEventId = null;
    state.seenMessageIds.clear();
    sessionStorage.removeItem(SESSION_KEY);
    timeline.replaceChildren();
    timelinePanel.hidden = true;
    enrollmentPanel.hidden = false;
    enrollmentTokenInput.value = "";
    channelInput.value = "";
    setEnrollmentMode(false);
    if (message) {
      showEnrollmentError(message);
    } else {
      enrollmentError.hidden = true;
      enrollmentError.textContent = "";
    }
  }

  function correctChannel(message) {
    stopStream();
    state.lastEventId = null;
    state.seenMessageIds.clear();
    timeline.replaceChildren();
    timelinePanel.hidden = true;
    enrollmentPanel.hidden = false;
    enrollmentTokenInput.value = "";
    channelInput.value = state.channelId || "";
    setEnrollmentMode(true);
    showEnrollmentError(message);
    channelInput.focus();
  }

  function formatTimestamp(value) {
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) {
      return "";
    }
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(date);
  }

  function appendMessage(message) {
    if (
      typeof message?.message_id !== "string" ||
      typeof message?.body !== "string" ||
      state.seenMessageIds.has(message.message_id)
    ) {
      return;
    }

    const wasNearBottom =
      timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 96;
    const fragment = messageTemplate.content.cloneNode(true);
    const item = fragment.querySelector(".message");
    const author = fragment.querySelector(".message-author");
    const time = fragment.querySelector(".message-time");
    const body = fragment.querySelector(".message-body");

    item.classList.toggle("agent", message.author_kind === "agent");
    item.dataset.messageId = message.message_id;
    author.textContent =
      typeof message.author_display_name === "string"
        ? message.author_display_name
        : "Unknown author";
    time.textContent = formatTimestamp(message.created_at);
    time.dateTime = typeof message.created_at === "string" ? message.created_at : "";
    body.textContent = message.body;

    state.seenMessageIds.add(message.message_id);
    timeline.append(fragment);
    emptyTimeline.hidden = true;
    if (wasNearBottom) {
      timeline.scrollTop = timeline.scrollHeight;
    }
  }

  async function authorizedFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("Authorization", `Bearer ${state.token}`);
    const response = await fetch(path, { ...options, headers, cache: "no-store" });
    if (response.status === 401) {
      throw new AuthenticationError("This Workshop session expired or was revoked.");
    }
    if (response.status === 403) {
      throw new ChannelAccessError("This session cannot access that Workshop channel.");
    }
    return response;
  }

  async function loadTimeline() {
    state.seenMessageIds.clear();
    timeline.replaceChildren();
    emptyTimeline.hidden = true;
    let cursor = null;
    let throughPosition = null;
    let pageCount = 0;

    do {
      const query = new URLSearchParams({ limit: "100" });
      if (cursor) {
        query.set("cursor", cursor);
      }
      const response = await authorizedFetch(
        `/v1/channels/${encodeURIComponent(state.channelId)}/timeline?${query}`,
      );
      const payload = await responsePayload(response);
      if (!response.ok) {
        throw new Error(safeErrorMessage(payload, "Could not load this channel."));
      }
      if (
        payload?.version !== 1 ||
        payload.channel_id !== state.channelId ||
        !Array.isArray(payload.messages) ||
        !Number.isSafeInteger(payload.through_position)
      ) {
        throw new Error("Kai returned an unsupported timeline response.");
      }
      if (throughPosition === null) {
        throughPosition = payload.through_position;
      } else if (throughPosition !== payload.through_position) {
        throw new Error("The timeline snapshot changed while it was loading.");
      }
      payload.messages.forEach(appendMessage);
      cursor = typeof payload.next_cursor === "string" ? payload.next_cursor : null;
      pageCount += 1;
      if (pageCount > 1000) {
        throw new Error("The timeline exceeded the client safety limit.");
      }
    } while (cursor);

    state.lastEventId = String(throughPosition);
    emptyTimeline.hidden = state.seenMessageIds.size !== 0;
    timeline.scrollTop = timeline.scrollHeight;
  }

  function parseEventBlock(block) {
    let eventName = null;
    let eventId = null;
    const data = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) {
        continue;
      }
      const separator = line.indexOf(":");
      const field = separator === -1 ? line : line.slice(0, separator);
      let value = separator === -1 ? "" : line.slice(separator + 1);
      if (value.startsWith(" ")) {
        value = value.slice(1);
      }
      if (field === "event") {
        eventName = value;
      } else if (field === "id") {
        eventId = value;
      } else if (field === "data") {
        data.push(value);
      }
    }
    if (!eventName) {
      return null;
    }
    return { eventName, eventId, data: data.join("\n") };
  }

  function applyTimelineEvent(event) {
    if (
      event.eventName !== "timeline.message.created" ||
      !/^\d+$/.test(event.eventId || "")
    ) {
      return;
    }
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }
    const eventPosition = Number(event.eventId);
    if (
      payload?.version !== 1 ||
      payload.channel_id !== state.channelId ||
      payload.message?.channel_id !== state.channelId ||
      payload.message?.event_position !== eventPosition
    ) {
      return;
    }
    appendMessage(payload.message);
    state.lastEventId = event.eventId;
  }

  async function consumeEventStream(generation) {
    state.controller = new AbortController();
    const headers = new Headers();
    headers.set("Last-Event-ID", state.lastEventId);
    const response = await authorizedFetch(
      `/v1/channels/${encodeURIComponent(state.channelId)}/events`,
      { headers, signal: state.controller.signal },
    );
    if (response.status === 409) {
      throw new ResynchronizationRequired();
    }
    if (!response.ok || !response.body) {
      const payload = await responsePayload(response);
      throw new Error(safeErrorMessage(payload, "Live updates are unavailable."));
    }

    setStatus("Live", "connected");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (generation === state.generation) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const event = parseEventBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (event) {
          applyTimelineEvent(event);
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
  }

  function delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  async function followEvents(generation) {
    while (generation === state.generation) {
      try {
        setStatus("Connecting live updates");
        await consumeEventStream(generation);
      } catch (error) {
        if (generation !== state.generation || error?.name === "AbortError") {
          return;
        }
        if (error instanceof AuthenticationError) {
          forgetSession(error.message);
          return;
        }
        if (error instanceof ChannelAccessError) {
          correctChannel(error.message);
          return;
        }
        if (error instanceof ResynchronizationRequired) {
          setStatus("Resynchronizing");
          try {
            await loadTimeline();
          } catch (resyncError) {
            if (resyncError instanceof AuthenticationError) {
              forgetSession(resyncError.message);
              return;
            }
            if (resyncError instanceof ChannelAccessError) {
              correctChannel(resyncError.message);
              return;
            }
            setStatus("Reconnecting", "disconnected");
            await delay(RECONNECT_DELAY_MS);
          }
          continue;
        }
        setStatus("Reconnecting", "disconnected");
      }
      await delay(RECONNECT_DELAY_MS);
    }
  }

  async function openStoredSession() {
    stopStream();
    const generation = state.generation;
    enrollmentPanel.hidden = true;
    timelinePanel.hidden = false;
    channelTitle.textContent = state.channelId;
    setStatus("Loading history");
    try {
      await loadTimeline();
      void followEvents(generation);
    } catch (error) {
      if (error instanceof AuthenticationError) {
        forgetSession(error.message);
      } else {
        correctChannel(error instanceof Error ? error.message : "Could not open this channel.");
      }
    }
  }

  enrollmentForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    enrollmentError.hidden = true;
    const channelId = channelInput.value.trim();
    const enrollmentToken = enrollmentTokenInput.value.trim();
    const deviceDisplayName = deviceNameInput.value.trim();
    if (!CHANNEL_PATTERN.test(channelId)) {
      showEnrollmentError("Enter the complete Workshop channel ID supplied by the operator.");
      return;
    }
    if (state.token) {
      state.channelId = channelId;
      saveSession();
      await openStoredSession();
      return;
    }
    if (!enrollmentToken || !deviceDisplayName) {
      showEnrollmentError("Device name, channel ID, and enrollment token are required.");
      return;
    }

    try {
      const response = await fetch("/v1/client/enrollment/redeem", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enrollment_token: enrollmentToken,
          device_display_name: deviceDisplayName,
        }),
        cache: "no-store",
      });
      const payload = await responsePayload(response);
      enrollmentTokenInput.value = "";
      if (!response.ok || typeof payload?.session?.token !== "string") {
        throw new Error(safeErrorMessage(payload, "Enrollment failed."));
      }
      state.token = payload.session.token;
      state.channelId = channelId;
      saveSession();
      setEnrollmentMode(true);
      await openStoredSession();
    } catch (error) {
      enrollmentTokenInput.value = "";
      showEnrollmentError(error instanceof Error ? error.message : "Enrollment failed.");
    }
  });

  forgetSessionButton.addEventListener("click", () => forgetSession());
  forgetEnrollmentSessionButton.addEventListener("click", () => forgetSession());
  window.addEventListener("beforeunload", stopStream);

  if (restoreSession()) {
    setEnrollmentMode(true);
    void openStoredSession();
  } else {
    setEnrollmentMode(false);
  }
})();
