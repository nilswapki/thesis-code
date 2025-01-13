import inspect
# allows import of CybORG class as:
# from CybORG import CybORG

from CybORG_plus.Debugged_CybORG.CybORG.CybORG.CybORG import CybORG

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/version.txt'
with open(path) as f:
    CYBORG_VERSION = f.read()[:-1]
