
# RGF-2 (v1)

An initial attempt to improve graph learning with a learned residual gate.

## Status

Version 1 is no longer under active development. The initial experiments did not provide enough evidence to determine whether the proposed gate improves robustness. The implementation issues and limitations identified during this work are documented here and in `docs/LESSONS.md`.

Development continues in the `rgf2` repository.

## Motivation

Graph neural networks learn node representations by aggregating information from neighboring nodes. This approach often works well when connected nodes share similar labels and the training and test data follow similar distributions. However, these conditions do not always hold, particularly in clinical and biological applications.

RGF-2 explored whether a learned residual gate could help address these challenges. The gate was placed between the residual and message-passing paths, allowing the model to adjust the contribution of each. The underlying idea was to reduce reliance on neighborhood information when it was unhelpful while preserving the original node representation when it was more informative.

The v1 implementation provided a working pipeline for exploring this idea, but the experiments were not sufficient to establish whether the approach worked as intended.

## Implementation

The v1 implementation was organized into seven Python modules:

```

src/rgf/
├── init.py
├── data.py      dataset loading and seeding
├── gate.py      residual gate
├── model.py     RGF layer and full model
├── train.py     training loop
├── eval.py      evaluation metrics
└── repro.py     seeding and configuration hashing

```

Each block combined a `GCNConv` layer with a residual gate. The gate took the current node representation and the message-passed representation as inputs, concatenated them, and produced feature-wise weights between zero and one. These weights controlled the contribution of the two paths to the block output.

The implementation also included a training and evaluation pipeline, random seeding, and configuration hashing. The pipeline completed successfully, and the experiment results were recorded.

## Experiment and Results

The first experiment used the Cora citation network. The model was trained for 200 epochs with a hidden dimension of 64, dropout of 0.5, and random seed 0.

| Split      | Accuracy | Macro F1 |
|------------|----------|----------|
| Train      | 1.000    | 1.000    |
| Validation | 0.756    | 0.747    |
| Test       | 0.749    | 0.733    |

A comparison was also attempted against a GCN baseline implemented in the same repository. That baseline achieved a test accuracy of 0.505, which is far below what a correct GCN implementation should produce.

The original GCN paper by Kipf and Welling reports 81.5 percent test accuracy on Cora using the public Planetoid split. The baseline in this repository did not follow the standard architecture: it added a linear layer before the graph convolutions and used GELU activations after each convolution. These changes alter the effective depth and non-linearity of the model and make it incomparable to the original architecture.

For this reason, the initial comparison does not establish that RGF-2 performs better than a standard GCN.

## What Went Wrong

Three problems became clear during the first experiment.

### The baseline was not a valid reference

The GCN implementation in this repository did not follow the architecture from Kipf and Welling (2017). It added a linear layer before the graph convolutions and used GELU activations after each convolution. These changes are not part of the standard formulation and make the baseline incomparable to published GCN results.

Because the purpose of the experiment was to assess whether the residual gate offers an advantage over an established graph neural network, the entire comparison was invalidated.

### The model overfit

Training accuracy reached 1.000 while validation accuracy remained at 0.756. This gap is consistent with memorization of the training nodes and suggests the learned representation does not generalize well to unseen nodes.

The experiment did not include enough additional evaluations to determine how much this affected the final results or which parts of the model contributed to it.

### The gate was never measured

The gate was the central component of the architecture, yet no measurements were taken of its output distribution. No ablation was run without it, no alternative initialization was tested, and its contribution to the final accuracy was never isolated.

Consequently, it remains unclear whether the gate learned a useful balance between the residual and message-passing paths. It is possible that its output remained close to a constant value, leaving the model with little benefit beyond ordinary message passing.

## Lessons Learned

The first version helped identify several issues that will guide the next set of experiments.

A reliable baseline comes first. Implementations should be checked against the original papers and evaluated using a documented protocol. Otherwise, differences in results may reflect implementation choices rather than the proposed method.

New components need to be evaluated directly. The presence of a learned gate does not, by itself, demonstrate that the gate improves the model. Its behavior should be measured, and its contribution should be tested through controlled ablation experiments.

One dataset cannot establish robustness. Cora is a useful starting point, but an experiment on a single citation network cannot establish that a model handles heterophily or distribution shift well. Those claims require evaluations designed specifically to test these conditions.

An inconclusive experiment can still be useful. The v1 results do not show that the residual-gating idea is ineffective. They show that the original experiment was not sufficient to determine whether it works. Identifying that limitation is an important step toward a more reliable evaluation.

## Repository Structure

```

rgf2-old/
├── src/rgf/           model and utilities
├── experiments/       experiment entry points
├── docs/
│   └── LESSONS.md     experiment limitations and lessons
├── .github/           continuous integration
├── requirements.txt
└── README.md

```

The repository structure was intentionally kept small. The initial focus was on implementing the model, running experiments, and recording the results.

## Related Work

### Standard graph neural networks

- Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017.
- Hamilton, Ying, and Leskovec, Inductive Representation Learning on Large Graphs, NeurIPS 2017.
- Veličković et al., Graph Attention Networks, ICLR 2018.

### Heterophily

- Zhu et al., Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs, NeurIPS 2020.
- Lim et al., Large-scale Learning on Heterophilic Graphs with a Two-stage Framework, 2021.

### Robustness and higher-order aggregation

- Zügner, Akbarnejad, and Günnemann, Adversarial Attacks on Neural Networks for Graph Data, KDD 2018.
- Abu-El-Haija et al., MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing, ICML 2019.

### Benchmarks

- Hu et al., Open Graph Benchmark: Datasets for Machine Learning on Graphs, NeurIPS 2020.

These papers provide reference points for implementing standard baselines, choosing evaluation datasets, and interpreting results in the context of published work.

## Next Steps

The next version will focus on establishing a reliable experimental foundation before making claims about the proposed architecture.

1. Implement and validate a standard GCN baseline following Kipf and Welling (2017).
2. Fix the evaluation protocol: identical splits, fixed seeds, documented hyperparameters, mean and standard deviation across runs.
3. Run an ablation study that isolates the contribution of the residual gate.

Once these steps are complete, the plan is to extend the evaluation to heterophilic datasets and to controlled distribution-shift settings. Details are documented in the v2 repository.

The goal of v2 is not simply to obtain better numbers. It is to determine whether the residual gate provides a measurable benefit, under which conditions that benefit appears, and where the approach falls short.

## License

MIT License.
```
