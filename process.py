# -*- coding: utf-8 -*-

import sys

infile = "app_name.dongliang.txt" #sys.argv[1]

lines = set()
lst = list()
try:
    fobj=open(infile,'r')
except IOError:
    print infile + ' open error:'
else:
    for eachLine in fobj:
        str = eachLine.strip()
        if str and str not in lines:
            lines.add(str)
    fobj.close

    for item in lines:
        lst.append(item)

    lst.sort(key=lambda x:len(x))

    try:
        fobj = open("out.txt", 'w')
    except IOError:
        print 'file open error:'
    else:
        for item in lst:
            fobj.write(item + '\r\n')
        fobj.close()
