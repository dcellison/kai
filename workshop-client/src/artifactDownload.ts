import type { WorkshopSession } from "./types";
import { ARTIFACT_PATTERN, CHANNEL_PATTERN } from "./types";

const DOWNLOAD_FRAME_LIFETIME_MS = 60_000;
let nextDownloadFrame = 0;

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

  nextDownloadFrame += 1;
  const frame = document.createElement("iframe");
  frame.name = `kai-artifact-download-${nextDownloadFrame}`;
  frame.hidden = true;
  frame.setAttribute("sandbox", "allow-downloads");

  const form = document.createElement("form");
  form.action = `/v1/channels/${encodeURIComponent(session.channelId)}/artifacts/${encodeURIComponent(artifactId)}/download`;
  form.method = "post";
  form.target = frame.name;
  form.hidden = true;

  const token = document.createElement("input");
  token.type = "hidden";
  token.name = "session_token";
  token.value = session.token;
  form.append(token);
  document.body.append(frame, form);

  try {
    form.submit();
  } catch (caught) {
    frame.remove();
    throw caught;
  } finally {
    form.remove();
  }
  window.setTimeout(() => frame.remove(), DOWNLOAD_FRAME_LIFETIME_MS);
}
