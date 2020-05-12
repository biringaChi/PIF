# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:39:49 2020

@author: rjsem
"""

#Need to pip install pytube

from pytube import YouTube

URL = "https://www.youtube.com/watch?v=nSdz5ln2rME"

yt = YouTube(URL)
yt = yt.get('mp4', '720p')

#possibl

yt.download('/path/to/download/directory')