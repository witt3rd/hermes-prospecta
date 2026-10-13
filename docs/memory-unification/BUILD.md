# Rebuild agent-memory-unification-v1

Everything that builds the PDF is in this folder:

| File | What it is |
|---|---|
| `document.md` | the report text (house `document` format: cover, contents, running header, numbered question cards) |
| `diagram.svg` | the one diagram (Figure 1) |
| `report.toml` | the pin (`publishing = "0.14.0"`), the project and the one fixed document entry |
| `agent-memory-unification-v1.pdf` | the built report (10 pages, letter) |

Needs `publishing` 0.14.0 (`uv tool install 'publishing[render] @ git+https://github.com/witt3rd/publishing@v0.14.0'`,
then `publishing setup` once for the pinned Chromium). The full evidence version of the same report,
with file references, is `../report.md`.

```sh
cd pdf
publishing build document.md --force    # rewrites agent-memory-unification-v1.pdf
publishing check document.md            # fails unless the PDF matches a fresh build
publishing build document.md --png /tmp/pages   # optional: one PNG per page to look at
```

The build fails, and writes nothing, on a layout problem, a glyph outside the vendored fonts, a secret,
a host detail (home path, e-mail), or a stale day word. That is why the text avoids words such as "today".
Edit `diagram.svg` and rebuild with `--force` (a change to the SVG alone does not mark the PDF stale).
