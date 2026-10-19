# REDUCED VARIANT of upstream savi/modules/__init__.py (repository
# google-research/slot-attention-video @ ba8f15ee19472c6f9425c9647daf87910f17b605).
#
# The upstream __init__ re-exports ~12 modules (convolution, resnet, pretrained, ...) that are not
# part of the pinned reference file set, so importing the package directly would fail.  This
# package marker is intentionally empty; reference runs import submodules explicitly, e.g.
# ``import savi.modules.attention``.
