#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crcnsget.py file obtained from neuromusic: https://github.com/neuromusic/crcnsget/blob/master/crcnsget/crcnsget.py
edited to add progress update, python3 syntax, and user-specified save file location
"""

# -*- coding: utf-8 -*-

import requests
from tqdm import tqdm

URL = 'https://portal.nersc.gov/project/crcns/download/index.php'

def download(datafile, username, password, filename):

    login_data = dict(
        username=username,
        password=password,
        fn=datafile,
        submit='Login' 
        )

    with requests.Session() as s:
        r = s.post(URL,data=login_data,stream=True)
        with open(filename, 'wb') as f:
            total_size_in_bytes= int(r.headers.get('content-length', 0))
            chunk_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
                progress_bar.update(len(chunk))
            progress_bar.close()