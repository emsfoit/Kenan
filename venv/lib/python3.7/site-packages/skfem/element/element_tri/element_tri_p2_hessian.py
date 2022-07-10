import numpy as np

from ..element_h1_hessian import ElementH1Hessian
from ...refdom import RefTri


class ElementTriP2Hessian(ElementH1Hessian):
    """Piecewise quadratic element with second derivatives."""

    nodal_dofs = 1
    facet_dofs = 1
    maxdeg = 2
    dofnames = ['u', 'u']
    doflocs = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.],
                        [.5, 0.],
                        [.5, .5],
                        [0., .5]])
    refdom = RefTri

    def lbasis(self, X, i):
        x, y = X

        if i == 0:
            phi = (1. - 3. * x - 3. * y + 2. * x ** 2
                   + 4. * x * y + 2. * y ** 2)
            dphi = np.array([-3 + 4. * x + 4. * y,
                             -3 + 4. * x + 4. * y])
            ddphi = np.array([
                [
                    4. + 0. * x,
                    4. + 0. * x,
                ],
                [
                    4. + 0. * x,
                    4. + 0. * x,
                ],
            ])
        elif i == 1:
            phi = 2. * x ** 2 - x
            dphi = np.array([4. * x - 1, 0. * x])
            ddphi = np.array([
                [
                    4. + 0. * x,
                    0. * x,
                ],
                [
                    0. * x,
                    0. * x,
                ],
            ])
        elif i == 2:
            phi = 2. * y ** 2 - y
            dphi = np.array([0. * x, 4. * y - 1])
            ddphi = np.array([
                [
                    0. * x,
                    0. * x,
                ],
                [
                    0. * x,
                    4 + 0. * x,
                ],
            ])
        elif i == 3:  # 0->1
            phi = 4. * x - 4. * x ** 2 - 4. * x * y
            dphi = np.array([4 - 8. * x - 4. * y, -4. * x])
            ddphi = np.array([
                [
                    -8. + 0. * x,
                    -4. + 0. * x,
                ],
                [
                    -4. + 0. * x,
                    0. * x,
                ],
            ])
        elif i == 4:  # 1->2
            phi = 4. * x * y
            dphi = np.array([4. * y, 4. * x])
            ddphi = np.array([
                [
                    0. * x,
                    4. + 0. * x,
                ],
                [
                    4. + 0. * x,
                    0. * x,
                ],
            ])
        elif i == 5:  # 0->2
            phi = 4. * y - 4. * x * y - 4. * y ** 2
            dphi = np.array([-4. * y, 4 - 4. * x - 8. * y])
            ddphi = np.array([
                [
                    0. * x,
                    -4. + 0. * x,
                ],
                [
                    -4. + 0. * x,
                    -8. + 0. * x,
                ],
            ])
        else:
            self._index_error()

        return phi, dphi, ddphi
