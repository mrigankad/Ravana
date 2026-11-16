# Contributing to Ravana

Thanks for helping. Small, focused pull requests get merged fastest.

## Set up

```bash
git clone https://github.com/mrigankad/Ravana.git && cd Ravana
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[gui,dev]"
ravana doctor                                     # checks GPU, onnxruntime and models
python -m pytest tests/ -q
```

Most tests run without model weights or a GPU. Tests that need weights skip themselves.

## Where to start

- Issues labelled [`good first issue`](https://github.com/mrigankad/Ravana/labels/good%20first%20issue).
- The roadmap and engineering plan in [`docs/research/`](docs/research/).
- UI work: the design kit in [`design/kit/KIT.md`](design/kit/KIT.md). Use the existing icons and Inter. No native dialogs or plain dropdowns.

## Rules

- **Consent first.** Don't add features whose main use is impersonation or non-consensual imagery. Demos and test images use consenting people or the repo's comparison set, never celebrities or politicians.
- **Model licences.** Code is MIT. If you add a model, state its licence in the model registry and in the README table.
- **Measure speed claims.** If a PR says "faster", include before and after numbers and the device.
- Format with `black` and `isort`. The CI checks both.

## Reporting bugs

Use the bug template and paste `ravana doctor --json`.
