# LearnerState — focused learning desk

What: an existing adaptive learning workspace and inspectable NLP pipeline.
Who: learners explaining studied concepts and researchers inspecting evidence.
Platform: responsive web, React/Vite, existing component architecture.
Goal: make the current question and next action immediately identifiable; expose failures and saved memory honestly.
Tone: clear, grounded, confident.

## Reference lock

Primary: [Grammarly DESIGN.md](https://styles.refero.design/style/be2af3f3-3886-4d70-973f-f7b5ab8d1a99), UUID `be2af3f3-3886-4d70-973f-f7b5ab8d1a99`. Adopt navy prose, deep teal actions, readable typography, flat borders, 8px corners and an 8px spacing rhythm.

Secondary: [Linear](https://styles.refero.design/style/90ce5883-bb24-4466-93f7-801cd617b0d1), UUID `90ce5883-bb24-4466-93f7-801cd617b0d1`. Borrow monospace technical metadata and compact, aligned diagnostic controls only. Fibery audit-log screenshot `2cec7602-2f2f-4abe-afc4-cdb4a03b094f` informs separated navigation and record inspection.

User-driven adaptation: use a midnight navigation rail to make sections visible, a warm gray canvas to distinguish white working panels, and larger, darker labels than the previous minimal theme. No marketing imagery or decorative gradients.

## Tokens and roles

- Ink `#182331`: headings and primary prose; secondary `#465566` on light surfaces.
- Navigation `#172b32`, text `#e8f0f1`, muted `#b9cbd0`; selected navigation `#294950` with a mint marker.
- Action `#086d60`, hover `#05564c`, white label. Teal denotes actionable controls and active selection.
- Canvas `#edf1ef`, panels `#ffffff`, quiet surface `#f4f7f5`, borders `#cdd8d3`.
- Warning text `#793d13` on `#fff5e7`; error text `#932e2e` on `#fff0ed`. Include labels; never rely on color alone.
- Body 15–16px/1.6, helper text at least 13px, page titles 30–34px/1.2 at weight 700, question 22–25px/1.5 at weight 600. System sans fallback avoids a network dependency.
- Spacing 8/16/24/32/40px. Panel radius 8px. Borders separate surfaces, no decorative shadows.

## Composition and behavior

Desktop: fixed-width navigation, dominant question column, secondary topic-memory inspector. Give the question panel a strong top border and distinct answer field. Put saved memory in its own clearly titled header and indicate cross-conversation reuse. On smaller screens, stack the inspector below the question without horizontal overflow.

Diagnostics: grouped filters, distinct execution list and selected detail, state/status labels, readable technical payloads. Never present a completed model call as successful question generation when validation failed.

Failure: show the concrete safe failure reason, preserve submitted answers and drafts, label generation progress, and offer material correction when the source is insufficient. Retrying generation and viewing traces must leave learner estimates and answer counts unchanged.

Verification: inspect desktop and mobile screenshots; check keyboard focus, contrast, wrapping, visible next question, persistent reload, error recovery, and the absence of browser errors.
