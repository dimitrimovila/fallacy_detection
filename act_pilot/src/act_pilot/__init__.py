"""Pilot for rewriting two Walton critical questions, by hand and with a depth-one ACT loop.

The package reads a configuration (``configs/pilot.yaml``), the items and the outer folds,
runs a reimplementation of the ACT loop of Grari et al. (arXiv:2509.26433) at depth one on
each binary boundary, and evaluates the original question, the hand-written rewrites, the
ACT selections and a few control questions against lexical baselines on the outer folds.
"""

__version__ = "0.1.0"
