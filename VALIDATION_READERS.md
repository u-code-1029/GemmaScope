# Image reader validation — 2026-10-15

Executed on macOS ARM64 with the app's Python 3.12 environment, Pillow, NumPy, and RosettaSciIO 0.15.0. These tests validate decoding and storage; they do not measure SEM/TEM detection accuracy or validate Windows execution.

` .gemmascope/venv/bin/python3 -m unittest discover -s tests -p test_storage.py -v `: **22 tests passed**. Temporary generated images cover RGB, 16-bit TIFF normalization, inclusive TIFF frames, real MRC stack decoding, MRC volume rejection, corrupt files, content deduplication, symlinks, transactional labels/metadata, source revision invalidation, missing sources, verified relinking, Windows/UNC path relinking on macOS, portable render/model paths, concurrent database access, cancellation, and atomic run JSON.

Additional real-format smoke test used small fixtures from the [official RosettaSciIO v0.15.0 test-data tree](https://github.com/hyperspy/rosettasciio/tree/v0.15.0/rsciio/tests/data). Each downloaded file matched its SHA-256 in the [official registry](https://github.com/hyperspy/rosettasciio/blob/v0.15.0/rsciio/tests/registry.txt). Files are local under `.cache/test/scientific-fixtures/`, excluded from distribution. Detailed measured output is `.cache/test/readers-validation.json`.

| Official fixture path | Measured result |
| --- | --- |
| `digitalmicrograph/2D/test-2.dm3` | One 2×2 frame decoded |
| `digitalmicrograph/2D/test-2.dm4` | One 2×2 frame decoded |
| `digitalmicrograph/3D/test-2.dm3` | Both 2×2 stack frames decoded with range 0–1 |
| `tia/old/64x64_TEM_images_acquire.emi` | One 64×64 image decoded; date 2016-02-21, magnification 19500, voltage 200 kV extracted |
| `tia/old/64x64_TEM_images_acquire_1.ser` | One 64×64 image decoded |
| `tia/old/16x16-point_spectrum-1x1024_1.ser` | Rejected as unsupported spectrum |

Aggregate: **6 frames imported, 1 unsupported file, 0 corrupt files**. SHA-256 checks before and after decoding confirmed every original fixture was unchanged. The EMI and its SER are separate source files, so source-content deduplication does not merge them.

Verified fixture SHA-256 values, in the table's order:

```text
f40485142df39dbdc06d7225983f3e0676c8a3a1caa614fbe5c1918c8a33b1c7
db23e0fa74d1c50ea1ef3c4df847e952aa15cc97eb25ea96deede46ba1e0edcd
b58bddb07242d3587b45166d5a7c6120079c9f3c3dfe69ac5c7e009eb8511981
d820ec205f047ed1db5e3a785e8f532e99ade7b70f65657aa184c6cce59f98eb
cfb0cb9c6ba5924a8a03db2d35c4864c5d3bb9a28ebe7c8a92710d0d8bcb9a98
118d531cfea1ac99e49a9cbdf17e8b6c254a6555a4772a58c9e7dca70d8954f8
```

Limits: vendor variants beyond these fixtures remain untested. RosettaSciIO 0.15.0 emits an upstream `ResourceWarning` for its MRC header file during Python unittest execution; the decode succeeds and our adapter never writes originals. Unsupported spectra, 4D data, annotated volumes, complex pixels, and ambiguous unannotated 3D signals are excluded. MRC image stacks require ISPG=0. Normalized RGB copies use per-frame min/max scaling for numeric scientific pixels and have a maximum dimension of 2048; this does not alter original quantitative data.
