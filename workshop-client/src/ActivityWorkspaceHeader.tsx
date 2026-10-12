import type { ReactNode } from "react";

import type { ConnectionState } from "./types";

export function ActivityWorkspaceHeader({
  actions,
  connection,
  symbol,
  title,
}: {
  actions?: ReactNode;
  connection: ConnectionState;
  symbol: ReactNode;
  title: string;
}): React.JSX.Element {
  // Same one-line bar as the conversation header: the title is styled like
  // the sidebar's "Kai Workshop" label and centered, with no breadcrumb.
  return (
    <header className="conversation-header activity-workspace-header">
      <h2>
        <span className="activity-heading-symbol" aria-hidden="true">{symbol}</span>
        {title}
      </h2>
      <div className="conversation-actions">
        {actions}
        <span className={`connection-indicator ${connection.tone}`} role="status">
          <span className="connection-dot" aria-hidden="true" />
          {connection.label}
        </span>
      </div>
    </header>
  );
}
