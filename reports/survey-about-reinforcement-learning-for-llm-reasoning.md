# A Survey on Reinforcement Learning for Large Language Model Reasoning

## TL;DR
- Reinforcement learning (RL) has transformed large language models (LLMs) from reactive token predictors into deliberative reasoners capable of multi-step planning, reflection, and self-correction [1].
- Algorithmic innovations like Group Relative Policy Optimization (GRPO) eliminate separate value networks, enabling scalable training on verifiable binary rewards [2][1].
- Advanced test-time compute paradigms integrate Process Reward Models (PRMs) and tree search algorithms (such as MCTS) to guide generation along high-probability logical trajectories [3][4][5].
- Recent breakthroughs like DeepSeek-R1 demonstrate that pure RL without SFT warmups can incentivize emergent long-horizon chain-of-thought and verification behaviors [1].
- Persistent challenges include the RLVR tax (overconfidence and reduced abstention), reward hacking, data contamination, and evaluation fragility [6][7][8].

## Background
Reinforcement learning for Large Language Model (LLM) reasoning bridges statistical next-token prediction and symbolic problem-solving. Early instruction tuning and Reinforcement Learning from Human Feedback (RLHF), epitomized by InstructGPT, relied heavily on Proximal Policy Optimization (PPO) with neural reward models trained on human preferences to align models with human values [9]. However, general human preference alignment often falls short in complex, multi-step formal domains like mathematics, coding, and logical deduction where answers possess objective ground truth. 

To address this, foundational paradigms shifted toward self-training and verifiable feedback. The Self-Taught Reasoner (STaR) framework demonstrated that LLMs could iteratively bootstrap their own reasoning capabilities by generating rationales, filtering for correct final answers, and performing supervised fine-tuning on successful trajectories [3]. Concurrently, preference optimization techniques such as Direct Preference Optimization (DPO) sought to streamline alignment by parameterizing rewards directly in terms of the language model policy, bypassing unstable online RL loops [10]. 

The transition from general conversational alignment to rigorous reasoning requires explicit control over generation length, search trajectories, and logical consistency. As models scale, standard autoregressive generation without test-time compute hits fundamental limits in recovering from early token errors [5]. Consequently, recent advances have married policy gradient optimization with verifiable binary rewards (RLVR) and structured search over reasoning states [5][1][11].

## Algorithmic Paradigms: From PPO to Value-Free Group Optimization
The computational bottleneck of traditional PPO in LLM training stems from maintaining a separate value network of comparable size to the policy model to estimate baseline advantages. To overcome this memory and compute overhead, DeepSeekMath introduced Group Relative Policy Optimization (GRPO) [2]. GRPO samples a group of outputs for each prompt, evaluates their rewards, and normalizes the scores within the group to estimate the baseline directly. This value-free design substantially reduces training memory while accelerating mathematical reasoning optimization [2][1].

Beyond policy gradient variants, iterative self-training loops have redefined how models acquire reasoning skills. STaR and subsequent methods like ReST-MCTS* integrate process-guided tree search to generate synthetic training data, alternating between exploration via search and policy updates [3][4]. Moreover, preference-based methods like DPO provide alternative avenues for aligning reasoning preferences without reward model exploitation [10]. However, recent empirical studies analyzing the optimization landscape ("Tricks or Traps?") show that hyperparameter choices—such as removing standard deviation terms in advantage normalization when reward distributions are highly concentrated on easy tasks—can dramatically stabilize training and prevent gradient explosion [6].

## Reward Modeling and Test-Time Search: ORMs versus PRMs
Supervision granularity is a central axis of differentiation in RL-driven reasoning. Outcome Reward Models (ORMs) provide sparse feedback based solely on the final answer or compiler verification [1]. While easy to automate using rule-based verifiers for math and code, sparse final-answer rewards become increasingly insufficient as reasoning trajectory length increases, leading to credit assignment failures [12]. 

To mitigate this, Process Reward Models (PRMs) evaluate intermediate reasoning steps [13]. Foundational works like Math-Shepherd established step-level verification datasets [13], but manual step annotations remain expensive. Recent breakthroughs address this via automated process supervision. Reasoning State Propagation (RSP) models prefix validity states across trajectories to guide beam search and RL [14]. Similarly, TIPS (Thinking-Induced Process Supervision) and FreePRM generate pseudo step-level labels from outcome-only reinforcement learning or buffer probability filtering, eliminating the need for manual annotations while outperforming fully supervised PRMs [15][16].

Integrating these reward signals with inference-time search further amplifies reasoning performance. Framing inference as search over a space of partial reasoning states unifies Monte Carlo Tree Search (MCTS), beam search, and best-of-N sampling [5]. When combined with process guidance, tree search allows models to dynamically backtrack from erroneous derivations, effectively scaling test-time compute [4][5]. However, directly employing ORMs or PRMs as RL training rewards can introduce severe reward hacking, where models generate redundant or convoluted steps to artificially inflate intermediate scores without solving the underlying problem [8].

## Recent Breakthroughs in Large-Scale Reasoning Models
The release of open and proprietary models has validated the efficacy of pure reinforcement learning at scale. DeepSeek-R1-Zero demonstrated that applying GRPO directly to base models without any supervised fine-tuning (SFT) phase allows advanced reasoning behaviors—such as self-reflection, verification, and dynamic strategy adaptation—to emerge organically [1]. To resolve issues like readability and language mixing, DeepSeek-R1 adopted a multi-stage pipeline combining cold-start long chain-of-thought examples with verifiable rule-based rewards (checking final answers via compilers and enforcing thinking tags) while intentionally avoiding neural reward models to prevent reward hacking [1].

Concurrently, the Qwen2.5-Math series and related RLVR (Reinforcement Learning with Verifiable Rewards) frameworks have shown that iterative self-improvement and rigorous verification significantly elevate mathematical competence [11][17]. However, training such large-scale reasoning models demands massive computational footprints and delicate reward signal tuning. Research into spurious rewards indicates that certain reinforcement signals can destabilize training or cause policy collapse if reward shaping is misconfigured across different model architectures [18].

## Evaluation Benchmarks, Challenges, and Open Problems
Despite impressive performance on benchmarks like MATH, GSM8K, and MBPP, the field faces significant methodological challenges. Recent studies highlight the "RLVR tax," wherein reinforcement learning with verifiable rewards leads to reduced abstention, overconfidence (shifting risk from appropriate uncertainty to assertive errors), decreased instruction fidelity on long generations, and expanded attack surfaces via lengthy reasoning traces [7]. Furthermore, ConsensusBench and related works emphasize that evaluation gaps and data contamination frequently inflate reported reasoning gains [12][7]. Under matched-budget and parity-controlled evaluations across multiple random seeds, several prominent performance gaps shrink or disappear, revealing that distributional sharpening can masquerade as genuine capability expansion [7].

Open problems also persist in balancing exploration and exploitation, mitigating reward hacking during dense reward shaping, and addressing the alignment tax where reasoning-optimized models exhibit degraded performance on general conversational tasks. Developing robust, contamination-resistant benchmarks and principled advantage estimation techniques remains critical for the next generation of reasoning LLMs [6][7].

## Trends and Open Problems
- **Shift to Value-Free and Group-Based Objectives:** The widespread adoption of GRPO and hybrid advantage estimators signals a departure from memory-heavy PPO architectures toward scalable, decentralized group optimization [2][6].
- **Automated Process Supervision:** Overcoming the bottleneck of manual step-level annotations via self-training frameworks (like TIPS and FreePRM) is democratizing process reward modeling [15][16].
- **Test-Time Compute Scaling:** Moving beyond single-trajectory generation to structured search (MCTS, tree search) is redefining how models expend compute during inference [4][5].
- **Mitigating the RLVR Tax and Overconfidence:** Addressing calibration failures, reduced abstention, and overconfidence in verifiable RL models remains an urgent open research area [7].
- **Robust Reward Design and Verifiability:** Eliminating reward hacking and spurious feedback loops is vital to ensuring that reasoning gains reflect true cognitive generalization rather than length exploitation [18][8].

## References
[1] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[2] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arxiv. https://arxiv.org/abs/2402.03300 (2024-02-05)
[3] STaR: Bootstrapping Reasoning With Reasoning. arxiv. https://arxiv.org/abs/2203.14465 (2022-03-28)
[4] ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search. arxiv. https://arxiv.org/abs/2406.03816 (2024-06-05)
[5] When LLM Meets Tree Search: A Systematic View of Inference as Search in Large Language Models. arxiv. https://arxiv.org/abs/2608.30395 (2026-08-31)
[6] Tricks or Traps? A Deep Dive into RL for LLM Reasoning. web. https://www.alphaxiv.org/abs/2508.08221 (2025-10-27)
[7] The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards. web. https://arxiv.org/html/2509.21882v1 (unknown)
[8] On Designing Effective RL Reward at Training Time for LLM Reasoning. web. https://arxiv.org/html/2410.15115 (2025-01-01)
[9] Training language models to follow instructions with human feedback. arxiv. https://arxiv.org/abs/2203.02155 (2022-03-04)
[10] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arxiv. https://arxiv.org/abs/2305.18290 (2023-05-29)
[11] Performance Foundations of Parallel & Distributed Reasoning Language Models. arxiv. https://arxiv.org/abs/2608.27046 (2026-08-27)
[12] ConsensusBench: Benchmark of Consensus Nodes for LLM Reasoning via Outcome Reward Densifying. arxiv. https://arxiv.org/abs/2609.04648 (2026-09-04)
[13] Math-Shepherd: Verify and Reinforce Language Models for Math Reasoning. arxiv. https://arxiv.org/abs/2310.02055 (2023-10-03)
[14] Learning Process Rewards via Reasoning State Propagation. arxiv. https://arxiv.org/abs/2609.39220 (2026-09-30)
[15] Inducing Process Supervision from Outcome-Only Reinforcement Learning. arxiv. https://arxiv.org/abs/2609.36641 (2026-09-29)
[16] FreePRM: Training Process Reward Models Without Ground Truth Process Labels. arxiv. https://arxiv.org/abs/2506.03570 (2025-06-04)
[17] Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement. arxiv. https://arxiv.org/abs/2409.12122 (2024-09-18)
[18] Spurious Rewards: Rethinking Training Signals in RLVR. hf-search. https://huggingface.co/papers/2506.10947 (2025-06-12)
