# LearnerState — focused learning desk

What: an existing adaptive learning workspace and inspectable NLP pipeline.
Who: learners explaining studied concepts and researchers inspecting evidence.
Platform: responsive web, React/Vite, existing component architecture.
Goal: make the current question and next action immediately identifiable; expose failures and saved memory honestly.
Tone: clear, grounded, confident.

## Reference lock

Primary: [Grammarly DESIGN.md](https://styles.refero.design/style/be2af3f3-3886-4d70-973f-f7b5ab8d1a99), UUID `be2af3f3-3886-4d70-973f-f7b5ab8d1a99`. Adopt navy prose, deep teal actions, readable typography, flat borders, 8px corners and an 8px spacing rhythm.

Secondary: [Linear](https://styles.refero.design/style/90ce5883-bb24-4466-93f7-801cd617b0d1), UUID `90ce5883-bb24-4466-93f7-801cd617b0d1`. Borrow monospace technical metadata and compact, aligned diagnostic controls only. Fibery audit-log screenshot `2cec7602-2f2f-4abe-afc4-cdb4a03b094f` informs separated navigation and record inspection.

Current composition follows Taste's **framed-sheet** app direction, seed **1793011816**, keeping the established palette and system typography. A light green-gray navigation rail recedes behind the working sheet. Actual question, evidence and snapshot records supply the visual identity; there are no decorative metrics or stock imagery. Taste governs composition, Impeccable supplies critique, UI/UX Pro Max supplies design-system guidance, and the installed motion-principles skill supplies Emil Kowalski's interaction guidance. UI/UX Pro Max's search script is absent; no generated recommendations from that script are claimed.

## Tokens and roles

- The implementation source of truth is `apps/web/src/tokens.css`.
- Ink `#202b2b`: headings and primary prose; secondary `#52625e` on light surfaces.
- Navigation and canvas `#f1f4f2`; selected navigation `#e4ebe7`, with a neutral text-weight change.
- Action `#086d60`, hover `#05564c`, white label. Teal denotes actionable controls and active selection.
- Panels `#ffffff`, quiet surface `#f7f9f7`, borders `#d5ded8`; control borders `#9eafa5`.
- Warning text `#793d13` on `#fff5e7`; error text `#932e2e` on `#fff0ed`. Include labels; never rely on color alone.
- Body 15–16px/1.6, helper text 13px, page titles up to 32px at weight 700, questions 22–25px/1.5 at weight 600. Segoe UI/system sans avoids a network dependency. IDs/code retain monospace; numeric columns use tabular figures.
- Spacing 8/16/24/32/40px. Panel radius 8px. Borders separate surfaces, no decorative shadows.

## Composition and behavior

Desktop: 248px navigation, dominant question column, 344px secondary topic-memory inspector, 1600px maximum content width. Use quiet borders and a distinct answer field. Saved memory has a clearly titled header and explicit cross-conversation provenance. Tablet and phone layout rules live in `workspace-theme.css`; verify 1440, 1024, 768, 390 and 320px. Use the existing question/memory switch on narrow screens and keep tables inside their own scroll containers.

Diagnostics: grouped filters, distinct execution list and selected detail, state/status labels, readable technical payloads. Never present a completed model call as successful question generation when validation failed.

Interaction: 44px primary controls, visible focus outlines, a skip link, Escape to dismiss the nonmodal guide and return focus to its trigger. Existing reduced-motion rules remain enabled. Frequent navigation uses immediate feedback; no decorative motion is added. Reports, diagnostics and math load on demand with readable pending states and an explicit reload path if a view bundle fails.

Failure: show the concrete safe failure reason, preserve submitted answers and drafts, label generation progress, and offer material correction when the source is insufficient. Retrying generation and viewing traces must leave learner estimates and answer counts unchanged.

Verification: inspect desktop and mobile screenshots; check keyboard focus, contrast, wrapping, visible next question, persistent reload, error recovery, and the absence of browser errors.
