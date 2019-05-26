from pyspark import SparkContext
from operator import add
import math

sc = SparkContext('local','App')
text_file = sc.textFile("file:///Users/lilichen/Desktop/project2/project2_data.txt")

# document count
count = text_file.count()

# for example: word - ("docX I like data science data")
# map: ((docX I), 1/5)     ((docX like), 1/5)     ((docX data), 1/5)    ((docX science), 1/5)    ((docX data), 1/5)
# reduce: (ex: key = (docX I)...) ((docX I), 1/5)     ((docX like), 1/5)     ((docX data), 2/5)    ((docX science), 1/5)
# keyBy: (I, ((docX I), 1/5))    (like, ((docX like), 1/5))     (data, ((docX data), 2/5))  (science, ((docX science), 1/5))     -- 1/5 and 2/5 is tf
a = text_file.flatMap(lambda word: (((word.split(" ")[0], a), (1.0/(len(word.split(" "))-1))) for a in word.split(" ")[1:])) \
                        .reduceByKey(lambda a, b: a+b).keyBy(lambda r: r[0][1])

# map: (docX I)     (docX like)     (docX data)    (docX science)   (docX data)
# distinct: (docX I)     (docX like)     (docX data)    (docX science)
# map: (I, 1)   (like, 1)  (data, 1)     (science, 1)
# reduce: (I, #)     (like, #)     (data, #)   (science, #)  how many documents include the word
b = text_file.flatMap(lambda word: ((word.split(" ")[0], a) for a in word.split(" ")[1:])) \
                        .distinct().map(lambda pair: (pair[1], 1)).reduceByKey(lambda a, b: a+b)

# join a and b: (I, (  ((docX I), 1/5),   #  )  )   (like, (((docX like), 1/5), #))   (data, (((docX data), 2/5), #))  (science, (((docX science), 1/5), #))
# map: (I, docX, 1/5 * log(count/#) )   -- log part is idf, the whole thing is tf*idf    (like, docX, 1/5 * log(count/#) )  (data, docX, 2/5 * log(count/#) )
c = a.join(b).map(lambda r: (r[0], r[1][0][0][0], r[1][0][1] * math.log10(count/r[1][1])))


# map: (I,  1/5 * log(count/#)^2 ) -- (I , (tf*idf)^2 ) ... ( data, 2/5 * log(count/#)^2 )
# reduce: (I, sum of (tf*idf)^2 for all I) ...
# map: (I, sqrt (sum of (tf*idf)^2 for all I) )
d = c.map(lambda r: (r[0], r[2]*r[2])).reduceByKey(lambda a, b: a+b).map(lambda r: (r[0], math.sqrt(r[1])))

# keyBy: (I, (I, docX, 1/5 * log(count/#) ))
# join: (I, (I, docX, 1/5 * log(count/#) ), sqrt (sum of (tf*idf)^2 for all I))
# map: (docX, (I, 1/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all I)))   (docX, (data, 2/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all data)))
# filter : keep all words start with gene_ and end with _gene 
c = c.keyBy(lambda r: r[0]).join(d).map(lambda r: (r[1][0][1], (r[0], r[1][0][2], r[1][1]))).filter(lambda r: (r[1][0].startswith('gene_') & r[1][0].endswith('_gene')))

# join: (docX,  ((I, 1/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all I))),  (I, 1/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all I)))     ----(key, ((x,x,x), (x,x,x)))
#       (docX,  ((I, 1/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all I))), (like, 1/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all like)))
#       (docX,  ((I, 1/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all I))), (data, 2/5 * log(count/#), sqrt (sum of (tf*idf)^2 for all data)))
#       ... All combination of word pairs
# filter: I< I, I<like, I < data, I< science ( delete one of the same pair ex: (I, like) and (like, I), and delete same word pair combination ex:(I,I)
# map: ((I, data), (1/5 * log(count/#) * 2/5 * log(count/#))/(sqrt (sum of (tf*idf)^2 for all I) * sqrt (sum of (tf*idf)^2 for all data))
# reduce: ((I,data)  similarity  (I, data))
# sort
c = c.join(c).filter(lambda r: (r[1][0][0] < r[1][1][0])).map(lambda r: ((r[1][0][0], r[1][1][0]), (r[1][0][1]*r[1][1][1])/(r[1][0][2]*r[1][1][2]))).reduceByKey(lambda a, b: a+b).sortBy(lambda r: -r[1])

c.saveAsTextFile("file:///Users/lilichen/Desktop/project2/output")
