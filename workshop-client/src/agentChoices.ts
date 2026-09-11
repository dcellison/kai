import type {
  WorkshopAgentCapability,
  WorkshopCollaborationOperation,
} from "./types";

export const AGENT_CAPABILITIES: {
  description: string;
  label: string;
  value: WorkshopAgentCapability;
}[] = [
  {
    description: "Create ordinary text responses.",
    label: "Text generation",
    value: "text_generation",
  },
  {
    description: "Expose bounded tool activity in the run inspector.",
    label: "Tool activity",
    value: "tool_activity",
  },
  {
    description: "Work within an already-authorized workspace.",
    label: "Workspace execution",
    value: "workspace_execution",
  },
  {
    description: "Accept images when the selected runtime supports them.",
    label: "Image input",
    value: "image_input",
  },
  {
    description:
      "Delegate bounded tasks to other active agents in a shared channel.",
    label: "Agent delegation",
    value: "agent_delegation",
  },
];

export const COLLABORATION_TOOLS: {
  description: string;
  label: string;
  value: WorkshopCollaborationOperation;
}[] = [
  {
    description: "Read bounded canonical conversation context.",
    label: "Context reading",
    value: "context_read",
  },
  {
    description: "Add or remove reactions on messages.",
    label: "Reactions",
    value: "reaction",
  },
  {
    description: "Publish visible progress updates while working.",
    label: "Progress updates",
    value: "progress_publish",
  },
  {
    description: "Reply inside an existing thread.",
    label: "Thread replies",
    value: "thread_reply",
  },
  {
    description: "Publish a bounded artifact with provenance.",
    label: "Artifacts",
    value: "artifact_publish",
  },
  {
    description: "Delegate bounded work to another active agent.",
    label: "Agent delegation",
    value: "agent_delegation",
  },
  {
    description:
      "Remain available for bounded observation in opted-in group channels.",
    label: "Standing participation",
    value: "standing_participation",
  },
];
