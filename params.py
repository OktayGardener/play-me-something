#!/usr/bin/env python

import sys
import yaml

params = yaml.load(sys.stdin)
print(' '.join('--%s=%s' % (key, value) for key, value in params.items()))
