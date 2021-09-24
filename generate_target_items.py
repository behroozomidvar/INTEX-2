# This code generates the target variants HOM and HET.

import psycopg2
import time
import datetime
import operator

dataset = "amazon" # "imdb" or "amazon"

def compare_values(val1, val2, attrib):
	sim = 0
	if attrib == "price" or attrib == "helpful_all" or attrib == "helpful_true":
		val1 = val1.replace("$","")
		val2 = val2.replace("$","")
		val1_float = float(val1)
		val2_float = float(val2)
		normalization_value = 95
		if attrib == "price":
			normalization_value = 350
		sim = max(0, 1.0 - (abs(val2_float - val1_float) / float(normalization_value)))	
	elif attrib == "title":
		val1_words = set(val1.split(" "))
		val2_words = set(val2.split(" "))
		intersect = len(val2_words.intersection(val1_words))
		uni = len(val2_words.union(val1_words))
		sim = float(intersect) / float(uni)
	elif attrib == "category":
		val1 = val1.replace("[","")
		val2 = val2.replace("[","")
		val1 = val1.replace("]","")
		val2 = val2.replace("]","")
		val1 = val1.replace("'","")
		val2 = val2.replace("'","")
		val1_cats = set(val1.split(", "))
		val2_cats = set(val2.split(", "))
		intersect = len(val2_cats.intersection(val1_cats))
		uni = len(val2_cats.union(val1_cats))
		sim = float(intersect) / float(uni)
	elif attrib in ("main_cat","genre", "actor", "director"):
		sim = 1 if val1 == val2 else 0
	elif attrib == "release":
		val1_year = int(val1)
		val2_year = int(val2)
		diff = abs(val1_year-val2_year)
		sim = 1.0 - (diff/2021)
	elif attrib == "reviewtime":
		val1_time = 0
		val2_time = 0
		if dataset == "amazon":
			val1_time = datetime.datetime.strptime(val1,"%m %d, %Y")
			val2_time = datetime.datetime.strptime(val2,"%m %d, %Y")
		else:
			val1_time = datetime.datetime.strptime(val1,"%d %B %Y")
			val2_time = datetime.datetime.strptime(val2,"%d %B %Y")
		diff = abs((val2_time-val1_time).total_seconds())
		if int(diff) > 604800:
			sim = 0
		else:
			sim = 1 - (diff / 604800)
	return sim

conn = psycopg2.connect("dbname="+dataset+"_reviews port=5432 user=behrooz password=212799")
cur = conn.cursor()

data = {} # keys are item pairs and values of dict of attributes
pairs = []

attrib_list = ["category", "helpful_all", "helpful_true", "reviewtime", "title", "main_cat", "price"]

if dataset == "imdb":
	attrib_list = ["helpful", "reviewtime", "director", "title", "release", "genre", "actor"]

for attrib in attrib_list:
	print("starting with ",attrib)
	query_string = "select t1.element_id, t2.element_id, t1.value, t2.value from temp_attribs t1, temp_attribs t2 where t1.name = '"+attrib+"' and t2.name = '"+attrib+"' and t1.element_id <> t2.element_id order by t1.element_id, t2.element_id;"
	cur.execute(query_string)
	query_results = cur.fetchall()
	print("result are out")
	for query_result in query_results:
		the_key = str(query_result[0]) + "-" + str(query_result[1])
		try:
			data[the_key][attrib] = compare_values(query_result[2], query_result[3], attrib)
		except:
			data[the_key] = {}
			for attrib_ in attrib_list:
				data[the_key][attrib_] = 0
			pairs.append(the_key)
			data[the_key][attrib] = compare_values(query_result[2], query_result[3], attrib)
	print("done with ", attrib)

print("number of verified pairs: ", len(pairs))

data_overall = {}
for key in data:
	overall = 0
	for attrib in attrib_list:
		overall += data[key][attrib]
	avg = overall / len(attrib_list)
	data_overall[key] = avg


res1 = sorted(data_overall.items(), key=operator.itemgetter(1))
res2 = sorted(data_overall.items(), key=operator.itemgetter(1),reverse=True)

print("**** HET: ****")

count = 0
for x in res1:
	if x[1] == 0:
		continue
	parts = x[0].split("-")
	print(parts[0])
	print(parts[1])
	count += 1
	if count == 100:
		break

print("**** HOM: ****")

count = 0
for x in res2:
	parts = x[0].split("-")
	print(parts[0])
	print(parts[1])
	count += 1
	if count == 100:
		break

