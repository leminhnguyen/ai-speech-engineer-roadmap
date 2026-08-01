"""Check the small, deterministic contract used by tutorial lessons."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = ROOT / "curriculum"
REQUIRED = (
    "docs/en.md",
    "assets/concept.svg",
    "code/main.py",
    "code/generate_figures.py",
    "code/tests/test_main.py",
    "notebook/lesson.ipynb",
)


def lesson_directories():
    return sorted(path for path in CURRICULUM.glob("*/*") if path.is_dir())


def main():
    errors = []
    lessons = lesson_directories()
    if not lessons:
        errors.append("No lesson directories found under curriculum/.")
    for lesson in lessons:
        for relative in REQUIRED:
            if not (lesson / relative).is_file():
                errors.append(f"{lesson.relative_to(ROOT)}: missing {relative}")
        docs = lesson / "docs/en.md"
        if docs.is_file() and not docs.read_text(encoding="utf-8").startswith("# "):
            errors.append(f"{lesson.relative_to(ROOT)}: docs/en.md must begin with an H1")
    if errors:
        print("Curriculum audit failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Curriculum audit passed: {len(lessons)} lesson(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
