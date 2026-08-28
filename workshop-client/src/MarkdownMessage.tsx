import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import type { WorkshopMessageMention } from "./types";

interface MarkdownPosition {
  end?: { offset?: number };
  start?: { offset?: number };
}

interface MarkdownNode {
  children?: MarkdownNode[];
  data?: {
    hName?: string;
    hProperties?: Record<string, unknown>;
  };
  position?: MarkdownPosition;
  type: string;
  value?: string;
}

function resolvedMentionPlugin(
  body: string,
  mentions: WorkshopMessageMention[],
): () => (tree: MarkdownNode) => void {
  const codePoints = Array.from(body);
  const positionedMentions = mentions.map((mention) => ({
    ...mention,
    endOffset: codePoints
      .slice(0, mention.start + mention.length)
      .join("").length,
    startOffset: codePoints.slice(0, mention.start).join("").length,
  }));
  return () => (tree: MarkdownNode): void => {
    const visit = (node: MarkdownNode): void => {
      if (!node.children) {
        return;
      }
      const nextChildren: MarkdownNode[] = [];
      for (const child of node.children) {
        const start = child.position?.start?.offset;
        if (child.type !== "text" || typeof child.value !== "string" || start === undefined) {
          visit(child);
          nextChildren.push(child);
          continue;
        }
        const end = start + child.value.length;
        const contained = positionedMentions.filter((mention) =>
          mention.startOffset >= start &&
          mention.endOffset <= end &&
          body.slice(mention.startOffset, mention.endOffset) ===
            child.value?.slice(
              mention.startOffset - start,
              mention.endOffset - start,
            )
        );
        if (contained.length === 0) {
          nextChildren.push(child);
          continue;
        }
        let cursor = 0;
        for (const mention of contained) {
          const relativeStart = mention.startOffset - start;
          if (relativeStart > cursor) {
            nextChildren.push({
              type: "text",
              value: child.value.slice(cursor, relativeStart),
            });
          }
          nextChildren.push({
            type: "strong",
            data: {
              hName: "span",
              hProperties: {
                className: ["message-mention", `message-mention-${mention.kind}`],
                "data-principal-id": mention.principalId,
              },
            },
            children: [
              {
                type: "text",
                value: child.value.slice(
                  relativeStart,
                  relativeStart + (mention.endOffset - mention.startOffset),
                ),
              },
            ],
          });
          cursor = relativeStart + (mention.endOffset - mention.startOffset);
        }
        if (cursor < child.value.length) {
          nextChildren.push({ type: "text", value: child.value.slice(cursor) });
        }
      }
      node.children = nextChildren;
    };
    visit(tree);
  };
}

const MARKDOWN_COMPONENTS: Components = {
  a: ({ children, href, node: _node, ...properties }) =>
    href ? (
      <a
        {...properties}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ) : (
      <span className="markdown-unsafe-link">{children}</span>
    ),
  img: ({ alt }) => (
    <span className="markdown-image-omitted">
      {alt ? `[Image: ${alt}]` : "[Remote image omitted]"}
    </span>
  ),
};

// Memoized on the single string prop, so a re-render of the enclosing
// view (every keystroke and resize pointermove re-renders the whole
// WorkshopView) never re-runs the remark parse for unchanged messages;
// that parse, multiplied by the timeline length, is the dominant render
// cost of the app.
export const MarkdownMessage = memo(function MarkdownMessage({
  body,
  mentions = [],
}: {
  body: string;
  mentions?: WorkshopMessageMention[];
}): React.JSX.Element {
  const mentionPlugin = useMemo(
    () => resolvedMentionPlugin(body, mentions),
    [body, mentions],
  );
  return (
    <div className="markdown-message">
      {/* Without rehype-raw, embedded HTML is rendered as inert source text. */}
      <ReactMarkdown
        components={MARKDOWN_COMPONENTS}
        remarkPlugins={[remarkGfm, mentionPlugin]}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
});
