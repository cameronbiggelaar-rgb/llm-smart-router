# LLM Smart Router — Architecture Diagrams

Generated with [archify](https://hermes-agent.nousresearch.com/docs) (`deliver`, quality profile `showcase`).
Every diagram passes **9/9 deliver checks** (0 errors, 0 warnings) and browser containment at
1440×900 / 1600×1000 / 1920×1080 / 2048×1320.

## The set

| Diagram | Type | Answers |
|---|---|---|
| [`router-architecture`](router-architecture.html) | architecture | What the system is made of, who calls whom, and where the evidence lives |
| [`router-request-lifecycle`](router-request-lifecycle.html) | sequence | The exact order of a single request from POST to logged row |
| [`router-decision-path`](router-decision-path.html) | workflow | How a request picks a model: gate → classify → tier → breaker → serve |

Open the `.html` files directly — they are self-contained (no network, no CDN). Each has
guided views; the architecture diagram additionally carries evidence refs pinned to a
specific commit of `cameronbiggelaar-rgb/llm-smart-router`.

## Source of truth

Each diagram is authored as JSON next to its render:

- `router-architecture.json` → `.html`
- `router-request-lifecycle.json` → `.html`
- `router-decision-path.json` → `.html`

Re-render after editing:

```bash
node ~/.hermes/skills/diagramming/archify/bin/archify.mjs deliver architecture \
  docs/diagrams/router-architecture.json docs/diagrams/router-architecture.html \
  --quality showcase --repo-root .
```

## Verified facts encoded here

These came from the code and the live system, not from prose:

- **Size gate** — compaction triggers at `(context_length − max_tokens) × 0.75`; with
  `context_length: 128000` that is **83,712** tokens.
- **Four routing lanes** — normal routing, rethink/rearchitect, failure escalation, `force_model`.
- **Escalation** — `empty_content` from the primary escalates to the next candidate; local
  Ollama is the terminal fallback.
- **B34/B35 prompt-cache identity** — the derived key anchors on the **first user turn**
  (not the system prompt); the generic `custom` provider profile is overridden so
  `supports_prompt_cache_key=True`, which is what finally put a real session id on the
  chat-completions wire.
- **Evidence** — exactly one `router_logs.db` row per attempt.
