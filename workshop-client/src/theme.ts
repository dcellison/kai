export const DEFAULT_WORKSHOP_THEME_ID = "atom-one-dark" as const;

export const WORKSHOP_THEME_CATALOG = [
  {
    colorScheme: "dark",
    displayName: "Atom One Dark",
    themeId: DEFAULT_WORKSHOP_THEME_ID,
  },
] as const;

export type WorkshopThemeId = (typeof WORKSHOP_THEME_CATALOG)[number]["themeId"];

const THEME_HINT_KEY = "kai.workshop.theme-hint.v1";
const THEMES_BY_ID = new Map<string, (typeof WORKSHOP_THEME_CATALOG)[number]>(
  WORKSHOP_THEME_CATALOG.map((theme) => [theme.themeId, theme]),
);

export function isWorkshopThemeId(value: unknown): value is WorkshopThemeId {
  return typeof value === "string" && THEMES_BY_ID.has(value);
}

export function applyWorkshopTheme(value: unknown, remember = true): WorkshopThemeId {
  const themeId = isWorkshopThemeId(value) ? value : DEFAULT_WORKSHOP_THEME_ID;
  const theme = THEMES_BY_ID.get(themeId);
  if (!theme) {
    throw new Error("The default Workshop theme is unavailable.");
  }
  document.documentElement.dataset.workshopTheme = themeId;
  document.documentElement.style.colorScheme = theme.colorScheme;
  if (remember) {
    try {
      sessionStorage.setItem(THEME_HINT_KEY, themeId);
    } catch {
      // The canonical server preference remains authoritative when storage is unavailable.
    }
  }
  return themeId;
}

export function restoreWorkshopThemeHint(): WorkshopThemeId {
  try {
    return applyWorkshopTheme(sessionStorage.getItem(THEME_HINT_KEY), false);
  } catch {
    return applyWorkshopTheme(DEFAULT_WORKSHOP_THEME_ID, false);
  }
}

export function clearWorkshopThemeHint(): void {
  try {
    sessionStorage.removeItem(THEME_HINT_KEY);
  } catch {
    // Clearing the DOM theme below is sufficient when storage is unavailable.
  }
  applyWorkshopTheme(DEFAULT_WORKSHOP_THEME_ID, false);
}
