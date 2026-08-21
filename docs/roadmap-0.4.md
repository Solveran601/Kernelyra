# Roadmap: 0.4.0a1

This document describes planned work, not functionality available in
Kernelyra 0.3.

The 0.4 foundation is intended to add a manifest-first data contract and
resource-aware training paths for text, image, and audio workloads. The target
design includes optional LoRA fine-tuning for compatible pretrained models,
user-provided tokenizers through Python and JSONL adapters, and native streaming
data preparation where it is appropriate.

Video and 3D are future plugin-contract targets. They will not be described as
built-in Kernelyra trainers until dedicated data decoders, model architectures,
metrics, tests, and reproducible benchmark evidence are shipped.

Every new backend or modality must document its supported hardware, optional
dependencies, input contract, quality metrics, licensing, and reproducible test
coverage before it is advertised in the main README.
