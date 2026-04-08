# Map Alignment by QR Code

## Generate QR codes

Generates QR images and a PDF. Adjust parameters in `qr_code_generator.py` for size and layout.

```bash
python qr_code_generator.py
```

## Align two maps

You need two completed maps (old and new). The script estimates the Sim(3) between them and aligns the new map to the old one. Parameters are documented in the code.

```bash
python map_alignment.py
```
