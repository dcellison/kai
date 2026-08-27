# Workshop theme catalog

Kai Workshop maps curated source palettes onto its semantic color-token
contract. The source palettes provide each theme's identity; a small number of
foreground tones are adjusted within the source hue family where the original
syntax color would not meet Workshop's WCAG contrast target as interface text.
No upstream code, fonts, images, or executable assets are bundled.

The source revisions below were recorded on 2026-08-27.

| Workshop themes | Authoritative source | Revision/version | License |
| --- | --- | --- | --- |
| Atom One Dark | [`atom/one-dark-syntax`](https://github.com/atom/one-dark-syntax) | `9c96f4454362267ac45322063e193ccf9d2debb1` | MIT, Copyright GitHub Inc. |
| Atom One Light | [`atom/one-light-syntax`](https://github.com/atom/one-light-syntax) | `d84579027410c576086dfca14d934c4bd74b0438` | MIT, Copyright GitHub Inc. |
| Dracula | [`dracula/dracula-theme`](https://github.com/dracula/dracula-theme) | `2985f660b04e6961b0887ffac2f8d3f35f431698` | MIT, Copyright Dracula Theme |
| Nord | [`nordtheme/nord`](https://github.com/nordtheme/nord) | `1cef71605416a222e57225b544540ce0fcec18d4` | MIT, Copyright Sven Greb |
| Solarized Dark and Light | [`altercation/solarized`](https://github.com/altercation/solarized) | `62f656a02f93c5190a8753159e34b385588d5ff3` (v1.0.0beta2) | MIT, Copyright Ethan Schoonover |
| Catppuccin Mocha and Latte | [`catppuccin/palette`](https://github.com/catppuccin/palette) | `07d02aa110ef9eb7e7427afca5c73ba9cf7f8ebd` | MIT, Copyright Catppuccin Org |
| GitHub Light Default, Dark Default, and Dark Dimmed | [`primer/github-vscode-theme`](https://github.com/primer/github-vscode-theme) and [`primer/primitives`](https://github.com/primer/primitives) | GitHub Theme `6.3.5` at `cd78e5e4e7bcf132a6f428ae0f32264bb1b729cf`; Primer Primitives `7.10.0` at `f82864eb33c37f8624704bd996bc21b97d3c311b` | MIT, Copyright GitHub Inc. |

## Mapping policy

- Canvas, panel, inset, surface, and raised-surface tokens preserve the source
  palette's ordered background scale.
- Text and expressive tokens begin with the source palette's named foreground,
  accent, and syntax colors.
- When a named source color is too low-contrast for normal-sized interface text,
  Workshop uses a lighter dark-theme tone or darker light-theme tone in the
  same hue family. These adaptations are part of Kai's semantic mapping, not a
  claim that the adjusted value is an upstream palette constant.
- Borders and shadows are derived from the mapped foreground and background
  scales. They are structural UI treatments rather than source syntax colors.
- Every catalog entry defines the complete semantic token contract, declares
  its native `color-scheme`, and is checked automatically for text, status,
  focus, and primary-control contrast.
- Unknown or retired identifiers fall back to Atom One Dark in both canonical
  preference resolution and the pre-render browser hint.

The upstream projects retain their respective names, copyrights, and licenses.
Their licenses are permissive MIT licenses; the source repositories linked
above contain the complete license texts.
