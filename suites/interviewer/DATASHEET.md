# Interviewer suite datasheet

- **Purpose:** Test whether a voice model can draw out information through neutral, well-timed interviewing.
- **Status:** Development. The suite currently contains one worked example.
- **Scenario authoring:** Scenarios use fictional briefs and hidden stories. LLM-assisted drafts must receive independent review.
- **Splits:** Public scenarios live in `dev/`. Held-out scenarios live outside the public repository.
- **Sensitive content:** Some hidden stories include personal or health-related experiences. They are fictional, should stay non-graphic, and should not reward pressing for medical detail.
- **Known bias:** Scenarios reflect the authors' assumptions about interviews and consumer research. Review should check for stereotypes and uneven difficulty across personas.
- **Known limits:** English only. Personas and voices are simulated.
