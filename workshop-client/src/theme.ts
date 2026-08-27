export const DEFAULT_WORKSHOP_THEME_ID = "atom-one-dark" as const;

export const WORKSHOP_THEME_CATALOG = [
  {
    colorScheme: "dark",
    displayName: "Atom One Dark",
    themeId: DEFAULT_WORKSHOP_THEME_ID,
  },
  {
    colorScheme: "light",
    displayName: "Atom One Light",
    themeId: "atom-one-light",
  },
  {
    colorScheme: "dark",
    displayName: "Dracula",
    themeId: "dracula",
  },
  {
    colorScheme: "dark",
    displayName: "Nord",
    themeId: "nord",
  },
  {
    colorScheme: "dark",
    displayName: "Solarized Dark",
    themeId: "solarized-dark",
  },
  {
    colorScheme: "light",
    displayName: "Solarized Light",
    themeId: "solarized-light",
  },
  {
    colorScheme: "dark",
    displayName: "Catppuccin Mocha",
    themeId: "catppuccin-mocha",
  },
  {
    colorScheme: "light",
    displayName: "Catppuccin Latte",
    themeId: "catppuccin-latte",
  },
  {
    colorScheme: "light",
    displayName: "GitHub Light Default",
    themeId: "github-light-default",
  },
  {
    colorScheme: "dark",
    displayName: "GitHub Dark Default",
    themeId: "github-dark-default",
  },
  {
    colorScheme: "dark",
    displayName: "GitHub Dark Dimmed",
    themeId: "github-dark-dimmed",
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
