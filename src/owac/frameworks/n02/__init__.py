"""N02: a measurement-constrained local multi-hypothesis recurrent state field.

Design summary
--------------
A frame's measurements constrain a field of ``K`` competing latent explanations
``h[B,V,K,D,h,w]``. Each cycle re-scores every candidate against the *same*
frozen measurement set, so iterating improves the approximate solution without
inventing new evidence. New contradicting measurements mark the affected
(candidate, location) cells, the priority propagates one hop along recorded
dependency edges, and only the selected region is recomposed -- through a shared
ConvGRU cell plus a proposal head that rebuilds refuted cells from the current
residual.

What is borrowed and what is claimed:

* Borrowed material: ConvGRU-style recurrent cells, prediction-error
  organisation, multi-hypothesis filtering. These are established mechanisms.
* Candidate contribution under test: the *revision rule* -- contradiction
  localisation, dependency-propagated selection and local recomposition -- and
  whether it corrects a wrong explanation faster than ordinary recurrence at the
  same hypothesis count, cycle budget and history.

Nothing here is a verified claim; the mechanism experiments and the honest
negative results are in ``research/experiments/2026-10-12_n02_encoder.md``.

Modules
-------
``config``      configuration and the tiny/small/medium presets
``state``       measurement buffer, hypothesis field, revision bookkeeping
``measurement`` measurement prediction, support masks, robust likelihood/energy
``hypotheses``  initialisation, competition, proposal, update cell
``revision``    contradiction map, dependency propagation, local recomposition
``model``       the encoder that runs the cycles
``readout``     MAP and grouped hypothesis-token readouts plus the ambiguity probe
"""

from owac.frameworks.n02.config import LossWeights, MeasurementConfig, N02Config, RevisionConfig
from owac.frameworks.n02.model import N02Encoder, build_encoder
from owac.frameworks.n02.readout import (
    StructuredRepresentation,
    TokenReadout,
    hypothesis_tokens,
    map_hypothesis,
    two_candidate_probe,
)
from owac.frameworks.n02.state import HypothesisState, MeasurementBuffer, RevisionState

__all__ = [
    "HypothesisState",
    "LossWeights",
    "MeasurementBuffer",
    "MeasurementConfig",
    "N02Config",
    "N02Encoder",
    "RevisionConfig",
    "RevisionState",
    "StructuredRepresentation",
    "TokenReadout",
    "build_encoder",
    "hypothesis_tokens",
    "map_hypothesis",
    "two_candidate_probe",
]
