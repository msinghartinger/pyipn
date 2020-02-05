"""Summary
"""
from pathlib import Path
import numpy as np

from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv

from pyorbital.orbital import Orbital

from pyipn.io.package_utils import get_path_of_data_file, get_path_of_data_dir


def write_tle(st, sat_id, drange, name):
    """
Send query request to space-track.org for a specified timerange (drange) 
and satellite id and write the corresponding two-line-elements into a data file.

    Args:
        st (TYPE): Description
        sat_id (TYPE): Description
        drange (TYPE): Description
        name (TYPE): Description

    Returns:
        TYPE: Description
    """
    lines = st.tle(iter_lines=True, norad_cat_id=sat_id,
                   epoch=drange, format='tle')
    data_path = Path(__file__).resolve(
    ).parent.parent.parent.as_posix() + '/data/'+name+drange+'_tle.txt'

    with open(data_path, 'w') as fp:
        for line in lines:
            fp.write(line + "\n")

    return(Path(__file__).resolve().parent.parent.parent.as_posix() + '/data/'+name+drange+'_tle.txt')


def convert_to_decimal_days(dt):
    """Summary

    Args:
        dt (datetime): Description

    Returns:
        float: time in decimal day of year
    """
    dd = dt.timetuple().tm_yday + dt.hour/24 + dt.minute/(24*60) + \
        dt.second/(24*3600) + dt.microsecond/(24*3600*1000000)
    return dd


def find_closest_epoch(dt, tle_file):
    """
find closest epoch TLE to correstponding datetime from specified file and return the two lines

    Args:
        dt (datetime): time at which position is calculated
        tle_file (str): name of tle file

    Returns:
        str: returns two line elements with epoch closest to requested time
    """
    day = convert_to_decimal_days(dt)
    year = dt.year
    diff = []

    with open(get_path_of_data_file(tle_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            elem = line.split()
            if elem[0] == '1':
                epoch_day = float(elem[3][2:])
                epoch_year = 2000+int(elem[3][0:2])
                diff.append(abs((epoch_year-year)*365 + epoch_day - day))

        argmin = np.argmin(np.array(diff))
        line1 = lines[argmin*2]
        line2 = lines[argmin*2+1]

    return(line1, line2)


def position_skyfield(dt, tle_file):
    """
return position of satellite at time dt based on TLE file;
(apparently does inlcude deep space corrections;
https://github.com/skyfielders/python-skyfield/issues/142)
position in kilometers from the center of the earth (GCRS)

    Args:
        dt (datetime): time at which position is calculated
        tle_file (str): name of tle file

    Returns:
        TYPE: Description
    """
    line1, line2 = find_closest_epoch(dt, tle_file)

    satellite = twoline2rv(line1, line2, wgs72)
    position, velocity = satellite.propagate(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, (dt.second+0.000001*dt.microsecond))

    return position


def position_pyorbital(dt, tle_path, name):
    """
return position of satellite at time dt based on TLE file;
(no deep space)
position from the center of the earth (GCRS) (I believe in earth radii = 6378.135km)


    Args:
        dt (datetime): Description
        tle_path (str): Description
        name (str): Description

    Returns:
        TYPE: Description
    """
    line1, line2 = find_closest_epoch(dt, tle_path)

    satellite = Orbital(name, line1=line1, line2=line2)
    position, velocity = satellite.get_position(dt)

    return position
