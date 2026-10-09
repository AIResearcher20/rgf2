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
