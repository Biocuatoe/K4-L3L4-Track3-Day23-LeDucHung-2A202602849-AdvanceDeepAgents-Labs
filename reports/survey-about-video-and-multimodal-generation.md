# A Comprehensive Survey of Video and Multimodal Generation: Foundations, Architectures, and Open Challenges

## TL;DR
- Video and multimodal generation has transitioned from cascaded U-Net GANs to scalable Diffusion Transformers (DiTs) and autoregressive foundation models operating on compressed latent spatiotemporal patches [1][2].
- State-of-the-art frameworks like Sora, Open-Sora, and CogVideoX employ 3D causal VAE latent tokenizers and decoupled spatial-temporal attention mechanisms to scale video resolution and temporal consistency [3][2][4][5].
- Unified any-to-any architectures (such as VideoPoet, SwanSphere, and DreamX-Creator) integrate text, image, video, and audio generation within shared autoregressive-diffusion hybrid spaces [6][7][8].
- Comprehensive benchmarking suites like VBench, GenEval, and T2VSafetyBench reveal ongoing challenges in fine-grained compositional reasoning, physical simulation accuracy, and temporal safety [9][10][11].

## Background
Video and multimodal generation represents a cornerstone of modern generative artificial intelligence, aiming to synthesize realistic, temporally coherent, and semantically aligned dynamic visual and auditory content from textual, visual, or multi-modal prompts. Historically dominated by generative adversarial networks (GANs) and recurrent autoencoders, the field underwent a paradigm shift with the advent of diffusion models and large-scale Transformers [1]. Diffusion models established robust superiority in photorealistic synthesis and distribution coverage, while transformer-based sequence modeling enabled flexible multimodal conditioning. 

The importance of video and multimodal generation has surged due to its foundational role in building world simulators, interactive simulation environments, robotics policies, and content creation pipelines [2]. Foundational milestones include VideoPoet, which introduced a decoder-only transformer framework for zero-shot video generation across interleaved modalities [6], and Sora, which demonstrated that scaling diffusion transformers on spacetime latent patches yields emergent capabilities such as 3D consistency and rudimentary physics modeling [2]. Understanding these architectures is essential for navigating the transition from isolated text-to-video tools to unified any-to-any interactive ecosystems.

## Foundations and Architectural Paradigms
The foundational architecture of video generation models has evolved from discrete token-based autoregressive models to continuous diffusion transformers (DiTs). VideoPoet established an early precedent by employing a decoder-only transformer framework that treats video generation as next-token prediction over quantized visual codes, processing text, images, and videos jointly [6]. Concurrently, large-scale survey literature highlights how diffusion models have superseded GANs as the primary paradigm due to superior sample quality and stability [1].

To handle the immense computational cost of spatiotemporal sequences, modern systems rely heavily on latent space compression. For example, ARLON integrates asymmetric diffusion transformers with autoregressive models using a VQ-VAE tokenizer achieving 128x total compression (8x spatial and 16x temporal downsampling) in a 12-channel latent space [3]. Furthermore, efficient attention mechanisms such as Video DeltaNet (VDN) address the quadratic complexity of long video denoising by combining local softmax attention with bidirectional linear memory via Video Delta Attention (VDA) [12]. Detailed analyses of training dynamics, such as checkpoint-resolved censuses of Open-Sora STDiT models (ranging from 306M to 1.03B parameters), demonstrate how specific temporal-attention heads specialize during training to coordinate cross-frame coherence [13].

## Advanced Video Generation Methods and Scaled Synthesis
High-resolution, long-duration video synthesis requires specialized spatial-temporal modeling to avoid visual flicker and loss of semantic consistency. Sora pioneered the use of diffusion transformers operating directly on raw video compressed into lower-dimensional spacetime latent patches via 3D VAE networks, scaling effectively with compute to exhibit world-simulator properties [2]. Open-Sora 1.2 advanced democratization by introducing a 3D autoencoder with 4x temporal compression to process original frame rates, coupled with the Spatial-Temporal Diffusion Transformer (STDiT) which decouples spatial self-attention within frames and temporal attention across frames [5].

Other notable architectures include CogVideoX, which utilizes an expert transformer architecture combined with 3D causal VAE compression [4], and Latte, which explores diverse latent diffusion transformer variants across spatial and temporal dimensions [14]. For high-resolution efficiency, CascadeV implements a Wurstchen-inspired cascaded latent diffusion architecture with spatiotemporal alternating grid attention [15]. Additionally, techniques like DiTCtrl enable tuning-free multi-prompt longer video generation through attention control mechanisms in multi-modal diffusion transformers [2].

## Multimodal Generation, Audio-Visual Synthesis, and Cross-Modal Alignment
Extending beyond unimodal text-to-video, modern generative frameworks increasingly target unified any-to-any multimodal alignment and synchronized audio-visual synthesis. SwanSphere introduced a streaming synchronized spatial audio generation framework via autoregressive diffusion transformers to resolve the latency-quality tradeoff in generating audio directly from panoramic videos and text [7]. In parallel, distributed inference systems like vLLM-Omni address the serving challenges of any-to-any multimodal models that combine heterogeneous autoregressive LLMs and diffusion components [16].

Cross-modal benchmarks such as UniM evaluate unified interleaved multimodal learning across diverse generation tasks [17]. For native audio-video generation, DreamX-Creator leverages a compact 7B generator with cross-modal attention, reinforcement learning with multimodal feedback, and an autoregressive 2K refinement pipeline [8]. Comprehensive surveys on unified multimodal models classify these systems into diffusion-based, autoregressive-based, and hybrid architectures, emphasizing that tokenization strategy and cross-modal attention fusion remain central design axes [18].

## Benchmarks, Evaluation Metrics, and Open Challenges
Evaluating generative video models requires going beyond traditional image metrics like FID to capture temporal dynamics, physical plausibility, and semantic alignment. VBench established a hierarchical benchmark suite decomposing video generation quality into 16 disentangled dimensions spanning video quality and video-condition consistency, supported by tailored prompt suites [9]. For compositional alignment, GenEval provides an object-focused evaluation framework utilizing object detectors (Mask2Former) to assess spatial count, position, and color composition with high human agreement [10].

Safety and trustworthiness have become critical evaluation vectors. T2VSafetyBench established a taxonomy covering 12 critical aspects across 4 primary categories, exposing unique temporal risks where individual frames appear benign but the video sequence conveys harmful content, highlighting a persistent tradeoff between model usability and safety [11]. Complementary project initiatives like VBench++ extend these evaluations to adaptive image suites and trustworthiness dimensions [19]. Despite these advances, major open challenges remain, including prohibitive compute costs, limitations in accurate real-world physics simulation, and balancing open-source reproducibility with safety guardrails.

## References
[1] A Survey on Video Diffusion Models. web. https://dl.acm.org/doi/full/10.1145/3696415 (2024-11-07)
[2] Video generation models as world simulators. web. https://arxiv.org/abs/2402.17177v3 (2024-02-15)
[3] ARLON: Boosting Diffusion Transformers with Autoregressive Models for Long Video Generation. arxiv. https://arxiv.org/abs/2410.20502 (2024-10-27)
[4] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[5] Open-Sora: Democratizing Efficient Video Production for All. arxiv. https://arxiv.org/abs/2412.20404 (2024-12-29)
[6] VideoPoet: A Large Language Model for Zero-Shot Video Generation. arxiv. https://arxiv.org/abs/2312.14125 (2023-12-21)
[7] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer. arxiv. https://arxiv.org/abs/2605.30940 (2026-05-29)
[8] DreamX-Creator: Democratizing Native Audio-Video Generation at 2K Resolution. hf-search. https://huggingface.co/papers/2608.31106 (2026-08-31)
[9] VBench: Comprehensive Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2311.17982 (2023-11-29)
[10] GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. arxiv. https://arxiv.org/abs/2310.11513 (2023-10-17)
[11] Evaluating the Safety of Text-to-Video Generative Models (T2VSafetyBench). arxiv. https://arxiv.org/abs/2407.05965 (2024-09-08)
[12] Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation. arxiv. https://arxiv.org/abs/2609.20744 (2026-09-17)
[13] Temporal-Attention Head Specialization During Video Diffusion Training. arxiv. https://arxiv.org/abs/2609.31654 (2026-09-14)
[14] Latte: Latent Diffusion Transformer for Video Generation. hf-search. https://huggingface.co/papers/2401.03048 (2024-01-05)
[15] CascadeV: An Implementation of Wurstchen Architecture for Video Generation. hf-search. https://huggingface.co/papers/2501.16612 (2025-01-28)
[16] vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models. arxiv. https://arxiv.org/abs/2602.02204 (2026-02-02)
[17] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark. hf-search. https://huggingface.co/papers/2603.05075 (2026-03-05)
[18] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities. web. https://arxiv.org/html/2505.02567v6 (2025-05-02)
[19] Comprehensive Benchmark Suite for Video Generative Models (VBench Project Page). web. https://vchitect.github.io/VBench-project/ (2024-01-01)
