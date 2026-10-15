/**
 * Shared header action icons for the Workshop client.
 *
 * Header buttons across the three panes use the sidebar's `nav-tool-button`,
 * which sizes its SVG to fill the button. Text glyphs such as "←" or "×"
 * would follow the font size instead and render at a different size from
 * the drawn icons beside them, so the back and close actions are drawn as
 * SVGs here and reused wherever a header needs them.
 */

/** Left-pointing arrow for "Back to ..." header actions. */
export function BackIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path
        d="M19 12H5M11 6l-6 6 6 6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

/**
 * Chevron for the sidebar collapse toggle: points left to collapse the
 * expanded sidebar and right to expand the collapsed one. Drawn rather
 * than typed as "‹"/"›" because those glyphs sit off-center in their line
 * box, so they never lined up vertically inside the button.
 */
export function ChevronIcon({ direction }: { direction: "left" | "right" }): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path
        d={direction === "left" ? "M15 6l-6 6 6 6" : "M9 6l6 6-6 6"}
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

/** Cross for "Close ..." header actions. */
export function CloseIcon(): React.JSX.Element {
  return (
    <svg aria-hidden="true" fill="none" focusable="false" viewBox="0 0 24 24">
      <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}
