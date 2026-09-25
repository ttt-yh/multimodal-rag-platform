# Third-Party Data and Service Notices

This repository contains source code, metadata, evaluation annotations, and scripts. Large upstream documents, images, derived corpora, model outputs, index artifacts, and local evaluation reports are excluded from Git by default.

## TiDB Chinese documentation

The Chinese retrieval and visual evaluation cases were prepared from the public `pingcap/docs-cn` repository:

- Source: <https://github.com/pingcap/docs-cn>
- Pinned revision: `95e8751594c2c4881d60948342f81ecd25fb9382`
- Product branch: `release-8.5`
- Recorded upstream license: `CC-BY-SA-3.0`
- Copyright and trademarks remain with their respective owners, including PingCAP.

Per-file source URLs, revisions, hashes, and license fields are retained in `data/manifests/`. Evaluation annotations derived from these documents should be used and redistributed consistently with the upstream attribution and share-alike requirements.

## ViDoRe computer-science dataset

The data preparation plan references `vidore/vidore_v3_computer_science` at revision `d5cc75883d92e294f0c0fc2662551c9708a06ebc`. Its dataset card records `CC-BY-4.0`. The downloaded PDFs and parquet files are not committed to this repository; only reproducibility metadata and preparation scripts are retained.

## OmniDocBench

The parsing-validation plan references `opendatalab/OmniDocBench` at the pinned revision recorded in `data/manifests/sources.lock.json`. The captured dataset card did not provide a license value. OmniDocBench source files are therefore not redistributed in this repository. Users who download them must review the upstream terms independently.

## External model and parsing services

Qwen, Alibaba Cloud Model Studio/Bailian, MinerU, Chroma, PostgreSQL, Vue, FastAPI, LangChain-related terminology, and other product names are used only to identify compatible services or dependencies. They remain trademarks of their respective owners. API access is governed by each provider's terms and is not bundled with this repository.

## Project license status

This notice does not grant a license to the project's own source code. A repository-level `LICENSE` must be selected by the project owner before a final public `v1.0.0` release. Until then, the release should remain a candidate and ordinary copyright rules apply.
