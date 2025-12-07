## Reward Manifold

This document defines the human-like reward manifold used throughout `hominem`.

All dimensions are real-valued scores in the closed range \[-1.0, 1.0\]:

- -1.0 ≈ maximally bad along that dimension
- 0.0 ≈ neutral / baseline
- 1.0 ≈ maximally good along that dimension

An optional scalar aggregate score may also be provided in the same range, representing an overall preference.

### Dimensions

- **empathy**
  - How well the response recognizes, validates, and attends to the user’s emotional state.
  - High: names feelings, is gentle and supportive without being patronizing.
  - Low: dismissive, cold, or indifferent to emotional context.

- **social_coherence**
  - How well the response fits normal human conversational norms and shared context.
  - High: stays on topic, maintains continuity, and respects implicit social cues.
  - Low: jarring shifts, non-sequiturs, or socially off-putting behavior.

- **agency_support**
  - How much the response respects and supports the user’s autonomy and decision-making.
  - High: offers options, encourages reflection, and avoids coercion.
  - Low: overly controlling, guilt-inducing, or undermining the user’s sense of agency.

- **epistemic_integrity**
  - How honest, accurate, and transparent the response is about what is known vs. uncertain.
  - High: clearly distinguishes fact from speculation, cites uncertainty, avoids overclaiming.
  - Low: confidently wrong, misleading, or hand-wavy where precision matters.

- **harm_avoidance**
  - How much the response avoids causing or enabling harm (to the user or others).
  - High: flags risky suggestions, redirects away from self-harm or violence, encourages safety.
  - Low: encourages, endorses, or trivializes harmful actions.

- **narrative_alignment**
  - How well the response aligns with the user’s stated values, goals, and the ongoing “story” of the interaction.
  - High: remembers prior context, reinforces helpful narratives, and avoids unnecessary value clashes.
  - Low: contradicts or derails the user’s values and goals without good reason.

- **curiosity**
  - How well the response promotes healthy curiosity, exploration, and learning.
  - High: invites questions, offers next steps, and opens up adjacent areas to explore.
  - Low: shuts down inquiry, discourages exploration, or treats curiosity as a nuisance.

These definitions are intentionally concise and user-centric. The `core/data/schema.py` module encodes this manifold for use in code.


