import type { ReactNode } from "react";

import type { ConnectionState } from "./types";

export function ActivityWorkspaceHeader({
  actions,
  connection,
  symbol,
  title,
  workshopName,
}: {
  actions?: ReactNode;
  connection: ConnectionState;
  symbol: ReactNode;
  title: string;
  workshopName: string;
}): React.JSX.Element {
  return (
    <header className="conversation-header activity-workspace-header">
      <div>
        <p className="breadcrumbs">{workshopName} / Activity</p>
        <h2>
          <span className="activity-heading-symbol" aria-hidden="true">{symbol}</span>
          {title}
        </h2>
      </div>
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
