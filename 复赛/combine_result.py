import sys
from pyspark import SparkContext
import numpy as np

outputfile = sys.argv[1]
LR_idpath = sys.argv[2]
XGB_idpath = sys.argv[3]

sc = SparkContext(appName='test')
LR_id = sc.textFile(LR_idpath).collect()
XGB_id = sc.textFile(XGB_idpath).collect()
result = np.array(LR_id + XGB_id).astype(float).astype(int)
np.sort(result)
print 'Counts: '+str(len(result))
print result
sc.parallelize(result).saveAsTextFile(outputfile)
