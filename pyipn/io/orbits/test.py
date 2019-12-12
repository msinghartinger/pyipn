from pyipn.io.orbits import tle
from pyipn.geometry import Location

from math import sqrt
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, SphericalRepresentation
import numpy as np

#package to send query requests to space-track.org
from spacetrack import SpaceTrackClient
st = SpaceTrackClient('singhartinger.moritz@gmail.com', 'WhydoIneedsuchalongpassword')

import spacetrack.operators as op
import datetime as dt
drange = op.inclusive_range(dt.datetime(2018, 1, 1),
	                        dt.datetime(2018, 12, 31))

sat_id_GLAST = 33053
sat_id_INTEGRAL = 27540

#data_path_GLAST = tle.write_tle(st, sat_id_GLAST, drange, 'GLAST')
tle_path_G = Path(__file__).resolve().parent.parent.parent.as_posix() +'/data/GLAST2018-01-01 00:00:00--2018-12-31 00:00:00_tle.txt'
tle_path_I = Path(__file__).resolve().parent.parent.parent.as_posix() +'/data/INTEGRAL2018-01-01 00:00:00--2018-12-31 00:00:00_tle.txt'
GCRS_coord = tle.position_skyfield(dt.datetime.utcnow(), tle_path_G)
print(type(tle.position_pyorbital(dt.datetime.utcnow(), tle_path_G, 'GLAST')*6378.135))

Location.from_GCRS(GCRS_coord)



#data_path_INTEGRAL = tle.write_tle(st, sat_id_INTEGRAL, drange, 'INTEGRAL')
# pos1 = tle.position_skyfield(dt.datetime(2017, 12, 2, 9, 58, 00), data_path_INTEGRAL)
# print(-6371 + sqrt(pos1[0]**2 + pos1[1]**2 + pos1[2]**2))
# pos2 = tle.position_skyfield(dt.datetime(2017, 12, 5, 1, 49, 28), data_path_INTEGRAL)
# print(-6371 + sqrt(pos2[0]**2 + pos2[1]**2 + pos2[2]**2))
# pos3 = tle.position_skyfield(dt.datetime(2017, 12, 7, 17, 40, 15), data_path_INTEGRAL)
# print(-6371 + sqrt(pos3[0]**2 + pos3[1]**2 + pos3[2]**2))
# pos4 = tle.position_skyfield(dt.datetime(2017, 12, 10, 9, 30, 1), data_path_INTEGRAL)
# print(-6371 + sqrt(pos4[0]**2 + pos4[1]**2 + pos4[2]**2))
# pos5 = tle.position_skyfield(dt.datetime(2017, 12, 13, 1, 19, 37), data_path_INTEGRAL)
# print(-6371 + sqrt(pos5[0]**2 + pos5[1]**2 + pos5[2]**2))
# pos6 = tle.position_skyfield(dt.datetime(2017, 12, 15, 17, 9, 35), data_path_INTEGRAL)
# print(-6371 + sqrt(pos6[0]**2 + pos6[1]**2 + pos6[2]**2))
# pos7 = tle.position_skyfield(dt.datetime(2017, 12, 18, 8, 59, 38), data_path_INTEGRAL)
# print(-6371 + sqrt(pos7[0]**2 + pos7[1]**2 + pos7[2]**2))
# print('\n')
# pos8 = tle.position_skyfield(dt.datetime(2017, 12, 3, 17, 54, 00), data_path_INTEGRAL)
# print(-6371 + sqrt(pos8[0]**2 + pos8[1]**2 + pos8[2]**2))
# pos9 = tle.position_skyfield(dt.datetime(2017, 12, 6, 9, 44, 40), data_path_INTEGRAL)
# print(-6371 + sqrt(pos9[0]**2 + pos9[1]**2 + pos9[2]**2))
# pos10 = tle.position_skyfield(dt.datetime(2017, 12, 9, 1, 34, 53), data_path_INTEGRAL)
# print(-6371 + sqrt(pos10[0]**2 + pos10[1]**2 + pos10[2]**2))
# pos11 = tle.position_skyfield(dt.datetime(2017, 12, 11, 17, 34, 53), data_path_INTEGRAL)
# print(-6371 + sqrt(pos11[0]**2 + pos11[1]**2 + pos11[2]**2))
# pos12 = tle.position_skyfield(dt.datetime(2017, 12, 14, 9, 14, 51), data_path_INTEGRAL)
# print(-6371 + sqrt(pos12[0]**2 + pos12[1]**2 + pos12[2]**2))
# pos13 = tle.position_skyfield(dt.datetime(2017, 12, 17, 1, 4, 43), data_path_INTEGRAL)
# print(-6371 + sqrt(pos13[0]**2 + pos13[1]**2 + pos13[2]**2))
# pos14 = tle.position_skyfield(dt.datetime(2017, 12, 19, 16, 54, 32), data_path_INTEGRAL)
# print(-6371 + sqrt(pos14[0]**2 + pos14[1]**2 + pos14[2]**2))
# print(tle.position_pyorbital(dt.datetime.utcnow(), data_path_INTEGRAL, 'INTEGRAL')*6378.135)
