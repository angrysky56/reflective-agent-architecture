# Reflective Agent Architecture - References

## Core Theoretical Foundations

### Modern Hopfield Networks & Associative Memory

**Foundational Papers:**

1. **Hopfield-Fenchel-Young Networks (2024)**
   - Santos, S. J. R. D., Niculae, V., McNamee, D., & Martins, A. F. T.
   - "Hopfield-Fenchel-Young Networks: A Unified Framework for Associative Memory Retrieval"
   - arXiv:2411.08590
   - **Key Contribution**: Unifies traditional and modern Hopfield networks through energy minimization using Fenchel-Young losses
   - **Relevance to RAA**: Provides theoretical framework for Manifold component

2. **Modern Hopfield Networks with Continuous-Time Memories (2025)**
   - Santos, S. J. R. D., et al.
   - arXiv:2502.10122
   - **Key Contribution**: Compresses discrete Hopfield memories into continuous-time representations, reducing computational costs
   - **Relevance to RAA**: Efficiency optimization for Manifold implementation

3. **Input-driven Dynamics for Robust Memory Retrieval (2025)**
   - PMC 12017325
   - **Key Contribution**: External inputs reshape Hopfield energy landscape to drive memory retrieval under noise
   - **Relevance to RAA**: Mechanism for biasing Manifold retrieval with Pointer goals

4. **High-order Rotor Hopfield Neural Networks (2025)**
   - Chen, B., & Zhang, H.
   - Neurocomputing, Volume 616
   - **Key Contribution**: High-order connections improve storage capacity and noise robustness
   - **Relevance to RAA**: Potential enhancement for Manifold complexity

5. **Sparse Quantized Hopfield Network for Online-Continual Memory (2024)**
   - Nature Communications
   - **Key Contribution**: Online learning with local rules in continual settings
   - **Relevance to RAA**: Training strategy for Manifold

### Entropy-Based Metacognition & Uncertainty

**Entropy Monitoring:**

6. **ERGO: Entropy-guided Resetting for Generation Optimization (2025)**
   - UncertaiNLP Workshop, ACL 2025
   - **Key Contribution**: Uses predictive entropy to detect LLM confusion and trigger interventions
   - **Relevance to RAA**: Direct precedent for Director's entropy monitoring

7. **Semantic Energy: Detecting LLM Hallucination Beyond Entropy (2025)**
   - arXiv:2508.14496
   - **Key Contribution**: Semantic energy (Boltzmann) outperforms semantic entropy for uncertainty detection
   - **Relevance to RAA**: Potential improvement over Shannon entropy for clash detection

8. **Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification (2025)**
   - ACL Findings 2025
   - **Key Contribution**: Similarity-based methods using embeddings better capture semantic relationships
   - **Relevance to RAA**: Alternative approach to entropy monitoring

**Metacognitive AI:**

9. **Metacognitive Sensitivity: Key to Calibrating Trust with AI (2025)**
   - Fleming, S. M., et al.
   - PNAS Nexus, 4(5)
   - **Key Contribution**: Metacognition involves transforming uncertainty into propositional assessments
   - **Relevance to RAA**: Theoretical foundation for Director's monitoring function

10. **The Metacognitive Demands of Generative AI (2024)**
    - Tankelevitch, L., et al.
    - CHI 2024
    - **Key Contribution**: Framework for metacognitive monitoring and control in AI systems
    - **Relevance to RAA**: Design principles for Director component

11. **Adaptive Transformer Programs (2025)**
    - ICLR 2025
    - **Key Contribution**: Uncertainty-aware attention using Jensen-Shannon Divergence
    - **Relevance to RAA**: Dynamic attention mechanism for Processor

### Graph Neural Networks & Knowledge Representation

**Semantic Memory:**

12. **Modern Hopfield Networks for Graph Embedding (2022)**
    - Frontiers in Big Data
    - **Key Contribution**: Uses MHN memories as trainable parameters for network topology
    - **Relevance to RAA**: Potential hybrid Manifold approach (Hopfield + GNN)

13. **Semantic-guided Graph Neural Network for Heterogeneous Graph Embedding (2023)**
    - Expert Systems with Applications
    - **Key Contribution**: Jumping knowledge mechanism to address semantic confusion
    - **Relevance to RAA**: Alternative to pure Hopfield for Manifold

14. **Petri Graph Neural Networks (2025)**
    - Scientific Reports
    - **Key Contribution**: Higher-order multimodal interactions through Petri net formulation
    - **Relevance to RAA**: Advanced relational structure for Manifold

**Knowledge Graphs:**

15. **Practices, Opportunities and Challenges in Fusion of KGs and LLMs (2025)**
    - Frontiers in Computer Science
    - **Key Contribution**: Survey of KG-LLM integration approaches
    - **Relevance to RAA**: Context for Manifold as structured knowledge

## Implementation Resources

### Code Repositories

16. **Latent Structured Hopfield Network (LSHN)**
    - GitHub: [needs URL from search]
    - PyTorch implementation of continuous Hopfield dynamics in autoencoder
    - **Direct relevance**: Starting point for Manifold implementation

17. **hopfield-layers (PyTorch)**
    - PyTorch library for Modern Hopfield Networks
    - **Direct relevance**: Production-ready Manifold components

### Related Frameworks

18. **Workshop on Associative Memory @ICLR 2025**
    - https://nfam.vizhub.ai/
    - **Context**: Active research community and latest developments

19. **TensorFlow GNN 1.0 (2024)**
    - Google Research Blog
    - **Context**: Production-scale GNN implementation (if using hybrid approach)

## Philosophical & Neuroscience Context

20. **Bohm's Implicate Order**
    - Conceptual precedent for Manifold as "implicit order"

21. **Insight Problem Solving (Neuroscience)**
    - ACC monitors conflict (Director monitoring)
    - Hippocampus retrieves alternatives (Director search)
    - **Biological inspiration**: RAA mirrors human insight mechanisms

22. **Active Inference Framework**
    - Free energy minimization
    - Surprise → search for better model
    - **Theoretical parallel**: Entropy threshold = surprise signal

## Additional Reading

### System 1 vs System 2 Thinking

23. **Dual Process Theory**
    - Kahneman's fast/slow thinking
    - **Relevance**: RAA bridges associative (System 1) and metacognitive (System 2) processes

### Creativity & Restructuring

24. **Computational Theories of Creativity**
    - Conceptual blending, analogical reasoning
    - **Relevance**: RAA's "tunneling" as computational insight

---

## Citation Note

This is a working bibliography for the RAA research project. Full BibTeX entries to be added as implementation progresses.

Last updated: November 2025
