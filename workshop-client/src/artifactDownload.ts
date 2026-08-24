import type { WorkshopSession } from "./types";
import { ARTIFACT_PATTERN, CHANNEL_PATTERN } from "./types";

export function startArtifactDownload(
  session: WorkshopSession,
  artifactId: string,
): void {
  if (
    !CHANNEL_PATTERN.test(session.channelId) ||
    !ARTIFACT_PATTERN.test(artifactId) ||
    !session.token
  ) {
    throw new Error("Invalid artifact download authority.");
  }

  const form = document.createElement("form");
  form.action = `/v1/channels/${encodeURIComponent(session.channelId)}/artifacts/${encodeURIComponent(artifactId)}/download`;
  form.method = "post";
  form.target = "_self";
  form.hidden = true;

  const token = document.createElement("input");
  token.type = "hidden";
  token.name = "session_token";
  token.value = session.token;
  form.append(token);
  document.body.append(form);

  try {
    form.submit();
  } finally {
    form.remove();
  }
}
