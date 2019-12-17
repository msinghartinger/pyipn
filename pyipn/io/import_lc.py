"""Summary
"""
from pyipn.io.package_utils import get_path_of_data_dir
import datetime as dt
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta

import pdb

from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
import time
import wget
import sys
import h5py
import os


def read_integral_spiacs_file(filename, data_dir):
    """Summary
    
    Args:
        filename (str): Description
        data_dir (str): Description
    
    Returns:
        time (array<float>): in unix time seconds
        count (array<float>): event count
        trigime (astropy.time.Time): trigger time in utc
    
    """
    with open(data_dir+"/lc/INTEGRAL_SPIACS/"+filename, 'r') as f:
        lines = f.readlines()
        time_info = lines[0][13:].split()[0:2]
        day = int(time_info[0][1:3])
        month = int(time_info[0][4:6])
        year = int('20'+time_info[0][7:9])
        seconds = float(time_info[1])

        time = []
        count = []

        for line in lines[2:]:
            time.append(float(line.split()[0]))
            count.append(float(line.split()[1]))

    time = np.array(time)
    count = np.array(count)

    datetime = dt.datetime(year, month, day)
    seconds = dt.timedelta(seconds=seconds)
    datetime = datetime + seconds

    trigtime = Time(datetime, format='datetime', scale='utc')
    time = time + trigtime.unix

    return time, count, trigtime


def conv_to_utc_fermi(t):
    """Summary
    
    Args:
        t (float): time in seconds from tref
    
    Returns:
        astropy.time.Time: time in utc
    """
    # Fermi reference epoch in MJD, terrestrial time (tt)
    MJDREF = 51910.0007428703703703703
    tref = Time(MJDREF, format='mjd', scale='tt')
    tref = tref.utc

    time = tref + TimeDelta(t, format='sec')
    return time.utc


def read_gbm_trigdat_file(filename, data_dir):
    """Summary
    
    Args:
        filename (str): Description
        data_dir (str): Description
    
    Returns:
        tstart (astropy.time.Time): start time of each data bin in unix time seconds
        combinedchannels (array<float>): combined count rates of all 8 channels for each of the 14 GBM detectors
        rate (array<float>): individual rate for each channel and detector
        binsize (array<float>): bin size in seconds
        trigtime (astropy.time.Time): trigger time in utc
        trigtimesec (float): trigger time in seconds from reference time
        pos (array(3)<float>): three cartesian dimensions of satellite position in gcrs
    """
    with fits.open(data_dir+"/lc/GBM_TRIGDAT/"+filename) as trigdat:
        trigtimesec = trigdat["EVNTRATE"].header['TRIGTIME']

        start_t = trigdat["EVNTRATE"].data['TIME']
        stop_t = trigdat["EVNTRATE"].data['ENDTIME']

        rate = trigdat["EVNTRATE"].data['RATE']

        pos = trigdat["EVNTRATE"].data['EIC']

    trigtime = conv_to_utc_fermi(trigtimesec)
    start = conv_to_utc_fermi(start_t[0])

    combinedchannels = np.transpose(np.sum(rate, axis=1))
    binsize = stop_t - start_t
    tstart = conv_to_utc_fermi(start_t).unix

    return tstart, combinedchannels, rate, binsize, start, trigtime, trigtimesec, pos


def read_gbm_tte_file(path):
    """Summary
    
    Args:
        path (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    with fits.open(path, memmap=True) as tte:
        event_t = tte["EVENTS"].data["TIME"]
        trigtimesec = tte["EVENTS"].header['TRIGTIME']

    tevent = conv_to_utc_fermi(event_t).unix
    trigtime = conv_to_utc_fermi(trigtimesec)

    return tevent, trigtime


def get_gbm_trigdat_files(year, data_dir):
    """Summary
    
    Args:
        year (int): Description
        data_dir (str): Description
    
    Returns:
        (list<str>): list of file names
    """
    url_start = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/" + \
        str(year)+"/"

    page = requests.get(url_start)
    data = page.text
    soup = BeautifulSoup(data)

    links = []

    for link in soup.find_all('a'):
        links.append(link.get('href'))

    re_bn = re.compile("bn.*")
    bnlinks = list(filter(re_bn.match, links))

    if not os.path.exists(data_dir+"/lc/GBM_TRIGDAT/"):
        os.makedirs(data_dir+"/lc/GBM_TRIGDAT/")

    trigdatfiles = []

    for idx, bnlink in enumerate(bnlinks):
        sys.stdout.write("\r{} of {}".format(idx, len(bnlinks)-1))
        sys.stdout.flush()

        url = url_start+bnlink+"current/"

        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data)

        links = []

        for link in soup.find_all('a'):
            links.append(link.get('href'))

        re_trigdat = re.compile("glg_trigdat.*")
        trigdatfile = list(filter(re_trigdat.match, links))

        if len(trigdatfile) > 0:
            trigdatfiles.append(trigdatfile[0])
            url = url + trigdatfile[0]
            wget.download(url, data_dir + "/lc/GBM_TRIGDAT/"+trigdatfile[0])
        else:
            print("No Trigdatfile found in: "+url)

        time.sleep(0.5)

    return trigdatfiles


def get_gbm_tte_files(year, data_dir):
    """Summary
    
    Args:
        year (TYPE): Description
        data_dir (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    url_start = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/" + \
                str(year)+"/"

    page = requests.get(url_start)
    data = page.text
    soup = BeautifulSoup(data)
    links = []

    for link in soup.find_all('a'):
        links.append(link.get('href'))

    re_bn = re.compile("bn.*")
    bnlinks = list(filter(re_bn.match, links))

    tot_ttefiles = []

    if not os.path.exists(data_dir+"/lc/GBM_TTE/"):
        os.makedirs(data_dir+"/lc/GBM_TTE/")

    for idx, bnlink in enumerate(bnlinks):
        sys.stdout.write("\r{} of {}".format(idx, len(bnlinks)))
        sys.stdout.flush()

        url1 = url_start+bnlink+"current/"

        page = requests.get(url1)
        data = page.text
        soup = BeautifulSoup(data)

        links = []

        for link in soup.find_all('a'):
            links.append(link.get('href'))

        re_tte = re.compile("glg_tte_.*")
        ttefiles = list(filter(re_tte.match, links))
        """
        make sure all existing directories are complete and not missing any files!
        """

        if os.path.exists(data_dir+"/lc/GBM_TTE/"+bnlink+"/"):
            continue
        else:
            os.makedirs(data_dir+"/lc/GBM_TTE/"+bnlink+"/")

        if len(ttefiles) > 0:
            tot_ttefiles.extend(ttefiles)
            for tte in ttefiles:
                url = url1 + tte
                wget.download(url, data_dir + "/lc/GBM_TTE/"+bnlink+"/"+tte)
        else:
            print("No TTE file found in: "+url1)

        time.sleep(300)

    return tot_ttefiles


def get_integral_spiacs_files(year, data_dir):
    """Summary
    
    Args:
        year (int): Description
        data_dir (str): Description
    
    Returns:
        (list<str>): list of file names
    """
    url_start = "https://www.isdc.unige.ch/integral/ibas/results/triggers/spiacs/"

    page = requests.get(url_start)
    data = page.text
    soup = BeautifulSoup(data)

    links = []

    for link in soup.find_all('a'):
        links.append(link.get('href'))

    re_y = re.compile(str(year)+"-../index\.....")
    ylinks = list(filter(re_y.match, links))

    comp_lc_urls = []
    lcfiles = []

    for ylink in ylinks:

        url = url_start+ylink

        page = requests.get(url)
        data = page.text
        soup = BeautifulSoup(data)

        links = []
        for link in soup.find_all('a'):
            links.append(link.get('href'))

        re_lc = re.compile(".*\.lc")
        for lc in list(filter(re_lc.match, links)):
            comp_lc_urls.append(url[:-10]+lc)
            lcfiles.append(lc)

    if not os.path.exists(data_dir+"/lc/INTEGRAL_SPIACS/"):
        os.makedirs(data_dir+"/lc/INTEGRAL_SPIACS/")

    for idx, u in enumerate(comp_lc_urls):
        sys.stdout.write("\r{} of {}".format(idx, len(comp_lc_urls)-1))
        sys.stdout.flush()

        wget.download(u, data_dir+"/lc/INTEGRAL_SPIACS/")

        time.sleep(0.5)

    return lcfiles


def get_reference_grb_position(year, data_dir):
    """
    get reference ra and dec positions of observed GRBs from 
    http://www.mpe.mpg.de/~jcg/grbgen.html and save them in hdf5 file
    
    Args:
        year (TYPE): Description
        data_dir (TYPE): Description
    """
    url_start="http://www.mpe.mpg.de/~jcg/grbgen.html"

    page = requests.get(url_start)
    data = page.text
    soup = BeautifulSoup(data)

    table = soup.find('table')
    rows = table.find_all('tr')
    links  = []

    for row in rows:
        link = row.find('a')
        if link != None:
            if int(link.string[0:2]) == (year%2000):
                links.append(link.get('href'))

    url_start = "http://www.mpe.mpg.de/~jcg/"
    grbs = []

    for idx, link in enumerate(links):
        sys.stdout.write("\r{} of {}".format(idx, len(links)-1))
        sys.stdout.flush()

        url = url_start + link
        subpage = requests.get(url)
        subdata = subpage.text
        soup = BeautifulSoup(subdata)
        grbtext = soup.find_all('pre')
        for text in grbtext:
            text = text.string
            if text != None:
                lines = text.splitlines()
                lines = {line.split()[0] : line.split()[1:] for line in lines if len(line)>1}
                
                try:
                    ra = float(lines['GRB_RA:'][0][:-1])
                    dec = float(lines['GRB_DEC:'][0][:-1])
                    date = lines['GRB_DATE:'][4]
                    time = lines['GRB_TIME:'][2][1:-1]

                except KeyError:
                    try:
                        ra = float(lines['EVENT_RA:'][0][:-1])
                        dec = float(lines['EVENT_DEC:'][0][:-1])
                        date = lines['EVENT_DATE:'][4]
                        time = lines['EVENT_TIME:'][2][1:-1]
                    except KeyError:
                        continue

                datetime = '20'+date+'_'+time
                time = Time.strptime(datetime, '%Y/%m/%d_%H:%M:%S.%f')

                grb = {'ra': ra, 'dec': dec, 'time': time}
                grbs.append(grb)

    with h5py.File((data_dir+"reference_grbs.hdf5"), 'a') as f:
        for grb in grbs:
            grp = f.require_group(grb['time'].strftime('%Y-%m-%d_%H:%M:%S.%f'))
            grp.attrs['time'] = grb['time'].strftime('%Y-%m-%d_%H:%M:%S.%f')
            grp.attrs['ra'] = grb['ra']
            grp.attrs['dec'] = grb['dec']


def integral_spiacs_to_hdf5(data_dir):
    """Summary
    
    Args:
        data_dir (str): Description
    """
    files = []
    for file in os.listdir(data_dir+"/lc/INTEGRAL_SPIACS/"):
        if file.endswith('.lc'):
            files.append(file)

    with h5py.File((data_dir+"/lc/INTEGRAL_SPIACS/integral_spiacs.hdf5"), 'a') as f:
        for filename in files:
            time, count, trigtime = read_integral_spiacs_file(
                filename, data_dir)
            grp = f.require_group(trigtime.strftime('%Y-%m-%d_%H:%M:%S.%f'))
            grp.attrs['trigtime'] = trigtime.strftime('%Y-%m-%d_%H:%M:%S.%f')
            try:
                del grp['time']
                t = grp.create_dataset('time', data=time)
            except KeyError:
                t = grp.create_dataset('time', data=time)
            try:
                del grp['count']
                c = grp.create_dataset('count', data=count)
            except KeyError:
                c = grp.create_dataset('count', data=count)


def gbm_trigdat_to_hdf5(data_dir):
    """Summary
    
    Args:
        data_dir (str): Description
    """
    files = []
    for file in os.listdir(data_dir+"/lc/GBM_TRIGDAT/"):
        if file.endswith('.fit'):
            files.append(file)

    with h5py.File((data_dir+"/lc/GBM_TRIGDAT/gbm_trigdat.hdf5"), 'a') as f:
        for filename in files:
            start, combinedchannels, rate, binsize, start_utc, trigtime, trigtimesec, pos = read_gbm_trigdat_file(
                filename, data_dir)
            grp = f.require_group(trigtime.strftime('%Y-%m-%d_%H:%M:%S.%f'))
            grp.attrs['trigtime'] = trigtime.strftime('%Y-%m-%d_%H:%M:%S.%f')
            grp.attrs['start_utc'] = start_utc.strftime('%Y-%m-%d_%H:%M:%S.%f')
            try:
                del grp['time']
                t = grp.create_dataset('time', data=start)
            except KeyError:
                t = grp.create_dataset('time', data=start)
            try:
                del grp['binsize']
                b = grp.create_dataset('binsize', data=binsize)
            except KeyError:
                b = grp.create_dataset('binsize', data=binsize)
            try:
                del grp['rate']
                t = grp.create_dataset('rate', data=rate)
            except KeyError:
                t = grp.create_dataset('rate', data=rate)
            try:
                del grp['combinedchannels']
                r = grp.create_dataset(
                    'combinedchannels', data=combinedchannels)
            except KeyError:
                r = grp.create_dataset(
                    'combinedchannels', data=combinedchannels)
            try:
                del grp['position']
                p = grp.create_dataset('position', data=pos)
            except KeyError:
                p = grp.create_dataset('position', data=pos)


def get_immediate_subdirectories(a_dir):
    """Summary
    
    Args:
        a_dir (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gbm_tte_to_hdf5(data_dir):
    """Summary
    
    Args:
        data_dir (TYPE): Description
    """
    tte_dir = data_dir+"/lc/GBM_TTE/"
    dirs = get_immediate_subdirectories(tte_dir)

    with h5py.File((data_dir+"/lc/GBM_TTE/gbm_tte.hdf5"), 'a') as f:
        idx = 0
        for a_dir in dirs:
            idx += 1
            sys.stdout.write("\r{} of {}".format(idx, len(dirs)))
            sys.stdout.flush()

            files = []

            for file in os.listdir(tte_dir+a_dir+"/"):
                files.append(file)

            all_tevent = np.empty(0, dtype=float)
            all_trigtime = []

            for filename in files:
                try:
                    tevent, trigtime = read_gbm_tte_file(tte_dir+"/"+a_dir+"/"+filename)
                except TypeError as te:
                    print(str(te)+" in file "+filename)
                    continue

                all_tevent = np.concatenate((all_tevent, tevent))
                all_trigtime.append(trigtime)

            sort_tevent = np.sort(all_tevent)

            grp = f.require_group(all_trigtime[0].strftime('%Y-%m-%d_%H:%M:%S.%f'))
            grp.attrs['trigtime'] = all_trigtime[0].strftime('%Y-%m-%d_%H:%M:%S.%f')
            try:
                del grp['tevent']
                t = grp.create_dataset('tevent', data=sort_tevent)
            except KeyError:
                t = grp.create_dataset('tevent', data=sort_tevent)
