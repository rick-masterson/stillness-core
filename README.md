# stillness-core

Prototype of the Stillness Core: a learning architecture that learns from lived
experience, with no stored datasets. Intent, decisions and results live in the
living design doc; this folder holds code only.

## Layout

- `stillness/field.py` - the field: dynamics that fall back to stillness, plus the gated plasticity rule (off when M = 0)
- `experiments/` - one script per rung of the test ladder

## Run

    pip install numpy matplotlib
    python3 experiments/rung0_stillness.py

## Ladder

- [ ] Rung 0: field returns to stillness (no learning)
- [ ] Rung 1: one-shot conditioning
- [ ] Rung 2: extinction
- [ ] Rung 3: generalization
- [ ] Rung 4: habituation
- [ ] Rung 5: voice prosody
- [ ] Rung 6: trajectory prediction
- [ ] Rung 7: second sense, no core changes
