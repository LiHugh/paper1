import logging
import numpy as np

logging.basicConfig(filename='practice_logging',level=10, filemode='w')
logging.debug('debug message')
logging.info('logging info message')
logging.warning('warn message')
logging.error('error message')

rg = np.random.RandomState(10)
print(rg)