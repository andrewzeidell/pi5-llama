# coding_standard.md

## Core Rules
- **Reuse before you write**: Search the `src` for needed functionality; import it when found.  
  - More instructions on the package contents of llm_utils is available in llm_utils_guide.md file
- Target latest `rust`; always annotate types.
- Keep functions small, pure, and idempotent when possible.
- Organize code:
  - `src/` main package & sub-packages
  - `tests/` rust test suite
  - `docs/` MkDocs material
  - `docs/uml/` PlantUML sources

