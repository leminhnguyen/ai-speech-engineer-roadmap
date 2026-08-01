# Lesson Template

Each lesson lives in `curriculum/NN-phase/NN-lesson-slug/` and contains:

```text
docs/en.md                 # learner-facing explanation
assets/concept.svg         # original conceptual diagram
assets/generated/*.png     # deterministic plots made by code/generate_figures.py
code/main.py               # self-contained runnable implementation
code/generate_figures.py   # figure generator
code/tests/test_main.py    # unittest coverage for core math
notebook/lesson.ipynb      # interactive companion; commit with cleared outputs
```

`docs/en.md` must include a title, metadata, learning objectives, intuition before implementation, a NumPy-first build, a library comparison, pitfalls, exercises, key terms, and primary references. Every diagram needs descriptive alt text and every code fence needs a language tag.
