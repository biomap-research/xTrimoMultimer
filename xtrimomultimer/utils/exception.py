class Error(Exception):
    """Base class for exceptions."""


class NoMatchTemplateError(Error):
    """An error indicating that cant find a precisely matched template."""


class TemplateAtomMaskAllZerosError(Error):
    """An error indicating that template mmCIF had all atom positions masked."""


class WrongAtomError(Error):
    """An error indicating that Found an unknown atom."""


class WrongPDBFormatError(Error):
    """An error indicating that Found an unknown coordinate."""


class WrongResError(Error):
    """An error indicating that Found an unknown res."""
