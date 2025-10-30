# Echo(I) Project Plan
## Learning Hierarchical Influence Propagation in Organizational Response Systems

**Team:** Hilal Hussain, Mahmoud Abdelmoneum, Theo Chen

## Project Overview

We formalize organizational response prediction as a Hierarchical Partially Observable Stochastic Game (H-POSG) where agents at different organizational levels make sequential decisions under uncertainty while influencing one another through a dynamic authority graph.

**Core Innovation:** Integrate game-theoretic formalization with deep learning (dual encoders + GAT + hierarchical policies) to predict both individual agent responses and collective organizational outcomes from text descriptions.

## Detailed Timeline (14 Weeks)

### Weeks 1-2: Infrastructure and Baselines
**Objective:** Foundation setup and simple baselines

- [ ] Environment setup (PyTorch, PyG, HuggingFace Transformers)
- [ ] Data structures for organizations, graphs, responses
- [ ] Implement bag-of-words + logistic regression baseline (target 40-50%)
- [ ] LLM-based generation setup for organizational structures
- [ ] Generate initial 1k organizations with 50k agent responses
- [ ] Implement rule-based Agent-Based Model (ABM)
- [ ] Data validation utilities and quality checks

**Deliverable:** 50k agent responses, documented ABM, baseline results

### Weeks 3-4: Core Model Development
**Objective:** Build dual encoders and initial GNN

- [ ] Implement dual encoders (BERT/RoBERTa) with contrastive training
- [ ] Basic 2-layer GAT implementation
- [ ] Integrate encoders + GNN for initial runs (target 55-60%)
- [ ] Extend GNN to 4 layers with temporal modeling (T=3 steps)
- [ ] Add hierarchical conditioning to policy network
- [ ] Integrate full pipeline (encoders → GNN → hierarchical policy)
- [ ] Hyperparameter search on validation set (target 65-70%)

**Deliverable:** Working end-to-end model, initial results

### First Project Break: Hypotheses and Experimental Design
- [ ] Analyze early results and refine approach
- [ ] Define testable hypotheses:
  - **H1:** Hierarchical cascades (higher-level → lower-level influence)
  - **H2:** Centralization accelerates consensus
  - **H3:** Culture-aligned recommendations → uniform support
  - **H4:** Power moderates peer influence
- [ ] Design controlled experiments for each hypothesis
- [ ] Expand dataset to 5k organizations (~350k responses)
- [ ] Finalize evaluation protocol

### Weeks 5-6: Scaled Training and Initial Experiments
**Objective:** Train on expanded data, test H1 and H2

- [ ] Train full model on 5k organizations
- [ ] Implement exploitability and stability metrics
- [ ] Run ablations: remove GNN/hierarchy/contrastive
- [ ] Visualize attention patterns over organizational charts
- [ ] Test H1: correlation between levels, cascade strength analysis
- [ ] Test H2: centralization metrics vs. consensus time
- [ ] Begin real-world data collection (business cases)

**Deliverable:** Ablation results, H1/H2 analysis

### Weeks 7-8: Advanced Experiments and Validation
**Objective:** Test H3, H4, validate on real cases

- [ ] Test H3: culture alignment vs. support uniformity
- [ ] Test H4: power level × peer influence interaction
- [ ] Cross-domain generalization tests (industries, sizes, cultures)
- [ ] Fine-tune on 500-1000 real organizational cases
- [ ] Validate against real cases, compute accuracy
- [ ] Analyze equilibrium approximation quality
- [ ] Sensitivity tests (perturbations to text, graph, features)
- [ ] Compare to MARL baselines (QMIX, independent Q-learning)

**Deliverable:** Hypothesis test results, real-world validation

### Second Project Break: Refinement and Analysis
- [ ] Identify best architectural variants (GNN depth, attention mechanism)
- [ ] Implement improvements: temporal transformers, multi-scale GNNs
- [ ] Comprehensive ablations on all components
- [ ] Analyze failure modes and edge cases
- [ ] Draft theoretical analysis section
- [ ] Generate final 10k-organization dataset (~750k responses)

### Weeks 9-10: Finalization
**Objective:** Final model training and complete evaluation

- [ ] Train final model on full 10k dataset
- [ ] Complete evaluation suite:
  - Predictive accuracy (individual + organizational)
  - Equilibrium properties (exploitability, Nash approximation)
  - Influence fidelity (attention-influence correlation)
  - Generalization (cross-domain, cross-size)
  - Real-world validation
- [ ] Implement interpretability analysis (SHAP, counterfactuals)
- [ ] Generate all visualizations (attention heatmaps, cascades, etc.)
- [ ] Synthesize results across all experiments
- [ ] Write final report (15-25 pages)
- [ ] Prepare presentation (20-30 slides)
- [ ] Package code, weights, and documentation
- [ ] Final quality checks and repository cleanup

**Deliverable:** Complete project with report, presentation, code

## Key Milestones
1. **Week 2:** Initial data pipeline + baseline (50k responses)
2. **Week 4:** Core model working end-to-end (65-70% accuracy)
3. **Week 6:** Scaled experiments + H1/H2 tested
4. **Week 8:** All hypotheses tested + real-world validation
5. **Week 10:** Final model + complete evaluation + deliverables

## Resources

### Computing
- 4x A100 GPUs (or equivalent)
- Estimated 48 hours training time for full model
- Mixed precision training (FP16)

### Datasets
- **Synthetic:** 10k organizations, 750k agent responses (LLM-generated)
- **Real:** 500-1000 business cases (HBR, Stanford, corporate reports)
- **ABM:** 100k simulation runs (rule-based validation)

### Key Literature
- Game Theory: Hansen et al. (2004) - POSGs, Jiang & Lu (2021) - Equilibrium refinements
- GNNs: Veličković et al. (2018) - GAT, Yang et al. (2021) - Information propagation
- MARL: Wen et al. (2021) - Actor-critic, Chen et al. (2025) - K-level policy gradients
- Dual Encoders: Liu et al. (2025) - Contrastive learning
- Organizational Theory: Hansen & Pigozzi (2024) - ABMs, Harrington (2012) - Agent-based orgs

## Team Responsibilities
- **Hilal:** Data generation, ABM, evaluation metrics
- **Mahmoud:** Model architecture, training pipeline, GNN implementation
- **Theo:** Dual encoders, experiments, visualization, documentation

## Risks & Mitigation

### Technical Risks
- **Risk:** Model convergence issues with multi-objective loss
  - *Mitigation*: Careful loss weight tuning, separate pre-training phases
  
- **Risk:** GNN depth causing vanishing gradients
  - *Mitigation*: Residual connections, layer normalization, gradient monitoring
  
- **Risk:** Synthetic data not representative of real organizations
  - *Mitigation*: Validate on real cases, fine-tune, use ABM for sanity checks
  
- **Risk:** Computational constraints (48h training × multiple experiments)
  - *Mitigation*: Mixed precision, efficient batching, distributed training

### Research Risks
- **Risk:** Hypotheses not supported by results
  - *Mitigation*: Multiple hypotheses, thorough analysis of negative results
  
- **Risk:** Model doesn't capture game-theoretic properties
  - *Mitigation*: Equilibrium regularization, baseline comparisons
  
- **Risk:** Poor generalization to unseen organizational types
  - *Mitigation*: Diverse synthetic data, cross-domain testing

