import numpy as np
import sys
from pyspark import SparkContext

outputfile = sys.argv[1]
fea_path = sys.argv[2]
LR_id_path = sys.argv[3]

def transData(line):
    listLine = line.split(' ')
    fea_str_list = [' {}:{:.8f}'.format(i + 1, float(listLine[i])) for i in range(len(listLine)-1)]
    fea_str_list = [str(int(float(listLine[len(listLine)-1])))] + fea_str_list
    fea_str_list = ' '.join(fea_str_list)
    
    return fea_str_list

sc = SparkContext(appName='test')
LR_id = np.array(sc.textFile(LR_id_path).collect()).astype(float)
LR_id_index = [int(ele)-1 for ele in LR_id]
fea_Data_rdd = sc.textFile(fea_path)
fea_Data_rdd_Dense = fea_Data_rdd.map(lambda line: transData(line))
out_fea_Data = fea_Data_rdd_Dense.zipWithIndex().filter(lambda ele: ele[1] not in LR_id_index)
outRdd = out_fea_Data.keys()
print 'blackSize:'+str(outRdd.count())
print outRdd.take(5)
outRdd.saveAsTextFile(outputfile)
