from pyipn.io.orbits.tle import position_pyorbital, position_skyfield
from pyipn.geometry import Location

from pathlib import Path
import datetime as dt

def test_position_pyorbital():
	tle = 'GLAST2018-01-01 00:00:00--2018-12-31 00:00:00_tle.txt'
	date = dt.datetime(2018, 6, 13)

	pos  = position_pyorbital(date, tle, 'GLAST')

def test_position_skyfield():
	tle_G = 'GLAST2018-01-01 00:00:00--2018-12-31 00:00:00_tle.txt'
	tle_I = 'INTEGRAL2018-01-01 00:00:00--2018-12-31 00:00:00_tle.txt'
	date = dt.datetime(2018, 6, 13)

	pos_G = position_skyfield(date, tle_G)
	pos_I = position_skyfield(date, tle_I)
	loc_G = Location.from_GCRS(pos_G)
	loc_I = Location.from_GCRS(pos_I)


