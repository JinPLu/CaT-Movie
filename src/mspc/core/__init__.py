"""Cross-domain foundations: shared contracts, registries and numeric helpers.

This package deliberately re-exports nothing. ``wan22_contract`` sits below
the config parser, while ``component_registry`` sits above it, so pulling
either through the package root would close an import cycle. Import the
submodule you need.
"""
