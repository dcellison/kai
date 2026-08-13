import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

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

export function MarkdownMessage({ body }: { body: string }): React.JSX.Element {
  return (
    <div className="markdown-message">
      {/* Without rehype-raw, embedded HTML is rendered as inert source text. */}
      <ReactMarkdown
        components={MARKDOWN_COMPONENTS}
        remarkPlugins={[remarkGfm]}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
}
