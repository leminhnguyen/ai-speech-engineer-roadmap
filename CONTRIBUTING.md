# 🤝 Contributing

Thanks for helping. Each pillar changes at a different pace, so each has its own rules.

## 🗺️ Roadmap

The roadmap holds knowledge that stays useful for years.

- Prefer primary papers and official documentation.
- Explain the learning objective and why the resource improves the sequence.
- Keep each section ordered oldest first, and mark only true essentials with ⭐.
- Work from the last year or two usually belongs in the SOTA tracker first.

## 🛠️ Tutorials

- Follow [tutorials/LESSON_TEMPLATE.md](tutorials/LESSON_TEMPLATE.md).
- Keep examples deterministic, and give every diagram descriptive alt text.
- Add the lesson to [tutorials/README.md](tutorials/README.md) and update the progress badges.

## 📡 SOTA Tracker

- Add an entry to [sota/data/entries.yaml](sota/data/entries.yaml). The file header documents every field.
- Check the inclusion criteria in [sota/README.md](sota/README.md#-how-entries-are-chosen).
- Run `make sota` and commit the YAML change with the regenerated pages. Never edit `sota/topics/` by hand.

## ✅ Before Opening a PR

```bash
make check                 # audit + test + sota-check
make figures notebooks     # when you changed a lesson
```

The same checks run in [GitHub Actions](.github/workflows/ci.yml) on every push and pull request.
