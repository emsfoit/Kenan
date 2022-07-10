import numpy as np

from .element_h1 import ElementH1
from .element_dg import ElementDG
from .element_line import ElementLinePp

class ElementSkeleton(ElementH1):

    def __init__(self, refdom, p):
        self.refdom = refdom
        self.elem = ElementDG(ElementLinePp(p))
        self.facet_dofs = self.elem.interior_dofs
        self.dofnames = ['u', 'u']
        self.doflocs = np.array([[.0, .0],
                                 [1., .0],
                                 [1., .0],
                                 [.0, 1.],
                                 [.0, .0],
                                 [.0, 1.]])

    def lbasis(self, X, i):
        if i in range(self.refdom.nfacets * self.facet_dofs):
            facet_ix = int(i / self.facet_dofs)
            if facet_ix == 0:
                coord = slice(0, 1)
            else:
                coord = slice(1, 2)
            basis_ix = i % self.facet_dofs
            return (
                (self.elem.lbasis(X[coord], basis_ix)[0]
                 * self.refdom.on_facet(facet_ix, X)),
                0. * X
            )
        self._index_error()
