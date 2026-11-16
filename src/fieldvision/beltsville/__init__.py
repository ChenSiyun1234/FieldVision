"""Models fitted on the organisers' Beltsville fields themselves.

Kept apart from fieldvision.model on purpose: the European models are trained and frozen
without any Beltsville label; anything in this package is fitted on Beltsville labels and is
validated only by leave-one-year-out on those same 44 field-years.
"""
