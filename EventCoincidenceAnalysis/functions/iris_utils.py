''' Add yyyymmdd coordinate'''
def _pt_date(coord, time):
    return coord.units.num2date(time)

def add_yyyymmdd(cube, coord, name='yyyymmdd'):
    """add a coordinate of the form YYYYMMDD to a cube."""
    if not cube.coords(name):
        #_pt_date = iris.coord_categorisation._pt_date
        iris.coord_categorisation.add_categorised_coord(cube, name, coord,
            lambda coord, x: '%s%2.2i%2.2i' % (_pt_date(coord, x).year, _pt_date(coord, x).month, _pt_date(coord, x).day))
        s