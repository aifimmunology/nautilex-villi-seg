# nautilex-villi-seg
2026 Nautilex hackathon challenge to segment individual villi from mouse ileum 10x Xenium spatial transcriptomics data.

> **Data reference:** Zhang et al., *Nature*, 2025 — ["Neuro-epithelial circuits promote sensory convergence and intestinal immunity"](https://www.nature.com/articles/s41586-025-09921-z)

![Challenge overview](images/challenge_overview.png)

## Challenge Overview

The small intestinal villus is a finger-like projection of the intestinal mucosa with a well-defined crypt-villus axis. Segmenting individual villi from spatial transcriptomics data would enable powerful analyses of villus-level heterogeneity, cell composition, and spatial gene expression gradients. This challenge asks participants to develop a method to automatically segment individual villi from 10x Xenium data.

---

## Desired Outcome

Generate a mapping of cells to individual villi. Acceptable outputs include:

- **Cell-to-villus map**: a table or file mapping cell barcodes to a villus ID that distinguishes individual villi at the single-cell level
- **Segmented polygon output**: polygon annotations (e.g., GeoJSON) delineating individual villi in pixel or micron space

---

## Approach

Feel free to be creative and attack the challenge as you wish, using the transcripts, the morphology images, or both!

---

## Data Download

Six Xenium output bundles are provided as `.zip` files from mouse ileum tissue sections. Download each file and unzip before use.

| # | Sample ID | Download |
|---|-----------|----------|
| 1 | TIS09474-001-001 | [output-XETG00195__0050316__TIS09474-001-001__20250129__221130.zip](https://pftini_mouse-219b3400-2a33-483a-a457-d5272e41d7db.storage.googleapis.com/AIFI-2025-11-21T03%3A02%3A10.637227232Z/output-XETG00195__0050316__TIS09474-001-001__20250129__221130.zip?x-goog-signature=49937df3155c8c5ab3246589594cb6e0c7b35b681b4507d5dfcef3b857a2818ade9eb93bd4a9a6ee30d5cb93d3a308bcacd1ee795849c1d95482d4fc414dba4112d80ceaa3699d08b101e38e470f771694a843ca6f55efe04858da9398d28663ffbdf5a21310623213f383a1bd6af49820ae4c76236b07b1f7332b2fb05329f334d3cf6387e282cae528c41d35112aded834a1f98435001ae4a49ff403ade12398af973766097693945ecac622acadb131da66c890dce0c6ab610cdff051a834b1f94f6d5d684976f5cc6000f0d07c01ffcf1772926cbfd50348dab4bc468ae6deb507776038efe06f08fe81a08b33cc86b00d86d2d78ae9180016719ebd3f4e&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=prod-ledger-data-sa%40core-security-internal.iam.gserviceaccount.com%2F20260326%2Fus-west1%2Fstorage%2Fgoog4_request&x-goog-date=20260326T021156Z&x-goog-expires=604800&x-goog-signedheaders=host) |
| 2 | TIS09471-001-001 | [output-XETG00195__0050316__TIS09471-001-001__20250129__221130.zip](https://pftini_mouse-219b3400-2a33-483a-a457-d5272e41d7db.storage.googleapis.com/AIFI-2025-11-21T03%3A02%3A05.106830155Z/output-XETG00195__0050316__TIS09471-001-001__20250129__221130.zip?x-goog-signature=5ecc3fac1a374637d901e1b08324190e7818f05e8f6cea47ef6727ee15db57943a2a6bb4f5364abcf3e2b0ad2d979743bc3ac5b747f89f8a4ec9606c7db92bdab121b4c900ad7185f9bfd049a44a13510a46df8b737b6ac7db05a17a0dab823cedc1faf9da2c945321066af2ccfc321e63ebfb74634c86aa15470ce7103f3298eeb62637376ae376876b3a48a4e1dec14b77c926252c0facd70e231be1c845c41a00996bad98308f95bc25da476958003ddb6531b804d9b635a4d65cf1e3996e76bf0c6e07ee9dbc89dde78d89aaf7ceb40f252f5651efa6194f2a24f44d97a25f6489c5fc000ca530c57313b39a3790f0b4e1b8db02f62e4a779a91feb9a3b6&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=prod-ledger-data-sa%40core-security-internal.iam.gserviceaccount.com%2F20260326%2Fus-west1%2Fstorage%2Fgoog4_request&x-goog-date=20260326T021212Z&x-goog-expires=604800&x-goog-signedheaders=host) |
| 3 | TIS09472-001-001 | [output-XETG00123__0051755__TIS09472-001-001__20250215__002423.zip](https://pftini_mouse-219b3400-2a33-483a-a457-d5272e41d7db.storage.googleapis.com/AIFI-2025-11-21T03%3A00%3A27.626892158Z/output-XETG00123__0051755__TIS09472-001-001__20250215__002423.zip?x-goog-signature=111df8907a838c820b90b5c85bbfded27b1266c3c0d4cfaffe3ce085e1b17c7642988625c3b83a5f5f2b742d4489a7e9b58a0ebde40e1944d1ad7a6bedb3da8e653bca02f59d8ed63334784a837942157f87647794c13e7f00627feb179401af4ff16fae49cc0b46af5a267ac6cba794e9872190c119ac312079e8b20ade5a183a5abfd820c04102e4055dade96d4e107532e490a4448345e31f174204511c5d66dec0fb2cda99e207ec66581defc03a39e868d8e9850143ca515f9a662e7a780e0b94b576ac6f45aa9fb6e3a4db8daa217b1577b63e99e6b64f428374c3a7c9f77504119b166d5512f12f38f12591ec530eb243773d99f14be14b0e169ef586&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=prod-ledger-data-sa%40core-security-internal.iam.gserviceaccount.com%2F20260326%2Fus-west1%2Fstorage%2Fgoog4_request&x-goog-date=20260326T021223Z&x-goog-expires=604800&x-goog-signedheaders=host) |
| 4 | TIS09475-001-001 | [output-XETG00123__0051755__TIS09475-001-001__20250215__002423.zip](https://pftini_mouse-219b3400-2a33-483a-a457-d5272e41d7db.storage.googleapis.com/AIFI-2025-11-21T03%3A00%3A32.684108047Z/output-XETG00123__0051755__TIS09475-001-001__20250215__002423.zip?x-goog-signature=3c155e728f8d3d6e570944e20f7de7b55e3f144a9f7d53801333f52a52221bf3fd6f9f610d83c535e623bcea4b9379ffda9c1416805e44ac47dcefc8dd26eb349392c52cc2777d798b9097ca8b49aa70da093202b3cf7bad182639a33c4b19656ef4ef9c1554f9b9ad4f9400293607899b9611e68e382d11d5b8caa195934211e9a70b3b7ad3d216b001e30883e0ce60846e1270d429e03dedfb88eeda6b84d874291e418c16ceb8a932117d18dc119e637ff6162b267f3934f3cf32f4d84168adc638840d0d6398ad932e8f888b3d51c1be0832bfc3c2d9cfd610316ad7dcddf8c7c5dfec3f4726cf02e4c5e4710d0ddb6b36c96e7bb5c645ebfabbd04adf10&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=prod-ledger-data-sa%40core-security-internal.iam.gserviceaccount.com%2F20260326%2Fus-west1%2Fstorage%2Fgoog4_request&x-goog-date=20260326T021230Z&x-goog-expires=604800&x-goog-signedheaders=host) |
| 5 | TIS09473-001-001 | [output-XETG00123__0051763__TIS09473-001-001__20250215__002423.zip](https://pftini_mouse-219b3400-2a33-483a-a457-d5272e41d7db.storage.googleapis.com/AIFI-2025-11-21T03%3A00%3A37.667955215Z/output-XETG00123__0051763__TIS09473-001-001__20250215__002423.zip?x-goog-signature=c89a62405e763a345d6e106337574450e0f5d73ca80dc9f4fad5a7def60d1ea5df194879d51715fa227d35b62b7696727a3353bb40b8feb45d4a8c516b71c2b4f682130609cae3d5b9c3338085768e715bc7f255a5c696bd5a531a9266681737b0220ae6b2cc44d1993178f3a005397bf3f015c66b495468aa26f05237b375f71b0ab06e423e6f60f89f9d0ddaff1664595580743ab422b9b578ee4f4c050bf1ed0dca75e2a5bc8f0d072127bf49fd755f69e242d2252115cddf163b7d138858bf85f3ddaacd6fda753c03e77a2aa014bfe2845a4501eb6b65f2d5ee796e3af1f3ec6ac434c83a236e8991fd430fef02ea5c1548fa23ce2e9193d6929626ee94&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=prod-ledger-data-sa%40core-security-internal.iam.gserviceaccount.com%2F20260326%2Fus-west1%2Fstorage%2Fgoog4_request&x-goog-date=20260326T021236Z&x-goog-expires=604800&x-goog-signedheaders=host) |
| 6 | TIS09476-001-001 | [output-XETG00123__0051763__TIS09476-001-001__20250215__002423.zip](https://pftini_mouse-219b3400-2a33-483a-a457-d5272e41d7db.storage.googleapis.com/AIFI-2025-11-21T03%3A00%3A42.694869641Z/output-XETG00123__0051763__TIS09476-001-001__20250215__002423.zip?x-goog-signature=b67be66a3518fe700df9ca70dea2a726c5875dd2fdc4ebdf5141908beb069aa9cf2766a8f620871b29c444824375c69c8934336e183f6f1a40ef4026a88380fed137474da0864e2ceca704247724137b2ebc18a6a431d42300a8c32904f3243b92492b414980a7c0fd3989bd8624fd8e8abe2ce0641c11a3c2719d663133e26f95e08e781448631f632b2fb121d643e4daf2266e1a8c933d879dd8609bae58db982bc4709f6b4d356ad4964790a49ec7ed30bd8ea6e8f65b954459d5df06d1f17be227747dc0a23c6df7f195c6cfa778e6b74af3c51aca9e330c9c6255fdab8306abf21ad9b2686a6e7eea0565b25f175f5aebe673eadad08109c4e21df24036&x-goog-algorithm=GOOG4-RSA-SHA256&x-goog-credential=prod-ledger-data-sa%40core-security-internal.iam.gserviceaccount.com%2F20260326%2Fus-west1%2Fstorage%2Fgoog4_request&x-goog-date=20260326T021243Z&x-goog-expires=604800&x-goog-signedheaders=host) |

> **Note:** Download links expire April 2, 2026. 

---

## Xenium Output Folder Structure

After unzipping, each bundle has the following structure:

```
output-XETG00195__0050316__TIS09474-001-001__20250129__221130/
├── cell_boundaries.parquet        # Polygon boundaries for each segmented cell
├── nucleus_boundaries.parquet     # Polygon boundaries for each segmented nucleus
├── cells.parquet                  # Per-cell metadata (location, area, etc.)
├── cells.csv.gz                   # Per-cell metadata (CSV format)
├── cells.zarr.zip                 # Cell-by-gene expression matrix (Zarr format)
├── cell_feature_matrix.h5         # Cell-by-gene expression matrix (HDF5 format)
├── cell_feature_matrix/           # Cell-by-gene expression matrix (MEX format)
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
├── transcripts.parquet            # Per-transcript locations (x, y, z, gene)
├── morphology.ome.tif             # DAPI morphology image (OME-TIFF)
├── morphology_focus/              # Multi-resolution morphology image (Zarr)
├── experiment.xenium              # Experiment metadata (open with Xenium Explorer)
└── ...
```

Key files for this challenge:
- **`transcripts.parquet`** — x/y coordinates of every detected transcript; useful for spatial density-based villus segmentation
- **`cell_boundaries.parquet`** / **`cells.parquet`** / **`cells.csv.gz`** — segmented cell locations, shapes, and per-cell metadata
- **`cell_feature_matrix.h5`** — cell-by-gene expression matrix in HDF5 format; easy to load with `scanpy.read_10x_h5()`
- **`morphology.ome.tif`** — DAPI image; villi are visible as tissue morphology

---

## Example Annotations

An example GeoJSON file with manually annotated individual villi for sample **TIS09474-001-001** is provided in this repository:

```
annotations/TIS09474-001-001_annotation.geojson
```

Each feature in the GeoJSON is a polygon outlining a single villus. This can be used as ground-truth for method development or evaluation. The coordinates are in **pixel space** (multiply by `0.2125` to convert to microns).

To view the annotations overlaid on the tissue, load `experiment.xenium` in Xenium Explorer and import the GeoJSON as a custom overlay.

---

## Xenium Explorer

Xenium Explorer is 10x Genomics' free desktop application for interactively visualizing Xenium data — including the morphology image, cell segmentations, transcript locations, and gene expression.

**Download Xenium Explorer 4:** https://www.10xgenomics.com/support/software/xenium-explorer/downloads

To open a dataset, launch Xenium Explorer and open the `experiment.xenium` file from any of the unzipped output folders.

### Importing the Example Annotations

To view the provided villus annotations overlaid on the tissue in Xenium Explorer:

1. Open `experiment.xenium` for sample **TIS09474-001-001** in Xenium Explorer
2. On the Annotations tab, import the provided geoJSON


