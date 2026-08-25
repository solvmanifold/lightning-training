# Static project brief

Build the dashboard-safe export from the project root:

```bash
python3 brief/build.py && python3 brief/validate.py
```

The command recreates `brief/dist/`, writes frozen evidence to `brief.json`,
generates the 800×450 WebP project card, and records the source revision and UTC
build time in `manifest.json`. The export needs no project service and contains
no external runtime dependencies.

The project card is generated in `build.py` from project-authored typography and
geometric forms; it does not contain third-party imagery.
