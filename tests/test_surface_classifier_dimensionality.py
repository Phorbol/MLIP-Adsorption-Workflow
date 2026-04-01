from ase.build import fcc111, molecule

from adsorption_ensemble.surface.classifier import SlabClassifier


def test_classifier_identifies_slab_via_isolation():
    clf = SlabClassifier()
    slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
    out = clf.classify(slab)
    assert out.is_slab is True
    assert out.normal_axis is not None


def test_classifier_rejects_molecule():
    clf = SlabClassifier()
    mol = molecule("H2O")
    out = clf.classify(mol)
    assert out.is_slab is False
    assert out.normal_axis is None

