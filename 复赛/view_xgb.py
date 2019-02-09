import sys
from pyspark import SparkContext

outputfile = sys.argv[1]
inputfile = sys.argv[2]

sc = SparkContext(appName='test')
rdd = sc.textFile(inputfile)
result = rdd.collect()
result_final = []
for index, ele in enumerate(result):
    temp = ele.split('\t')
    if float(temp[1]) < 0.7:
    	result_final.append(str(0.0))
    else:
        result_final.append(str(1.0))

result_rdd = sc.parallelize(result_final)
result_rdd.saveAsTextFile(outputfile)
