from topotein.features.topotein_neighborhood_calculator import TopoteinNeighborhoodCalculator
from topotein.features.topotein_complex import TopoteinComplex

def compute_neighborhoods(batch, neighborhoods):
    if isinstance(batch.sse_cell_complex, TopoteinComplex):
        nc = TopoteinNeighborhoodCalculator(batch.sse_cell_complex)
        batch.sse_cell_complex.calculator = nc
    else:
        raise NotImplementedError("NeighborhoodCalculator for toponetx support is removed")
    return nc.calc_equations(neighborhoods)





