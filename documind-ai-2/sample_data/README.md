# Sample data

`solar_handbook.pdf` is a tiny synthetic 3-page PDF bundled for demos (the rest of `*.pdf` is git-ignored so you never commit private documents by accident).

To try the app quickly, generate a small synthetic PDF from the evaluation dataset:

```bash
python - <<'PY'
import json, pymupdf
data = json.load(open("eval/dataset.json"))
name, pages = next(iter(data["documents"].items()))
doc = pymupdf.open()
for text in pages:
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=12)
doc.save(f"sample_data/{name}")
print("wrote", f"sample_data/{name}")
PY
```

Then upload `sample_data/solar_handbook.pdf` and ask, for example, "How much can shading reduce output?".
(Remove the `*.pdf` ignore rule exception in `.gitignore` only if you intentionally want to commit a sample.)
