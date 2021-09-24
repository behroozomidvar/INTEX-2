# This library handles queries over the Postgres database (connection and execution).

import psycopg2
import pandas as pd
import configuration

class DBInterface:
    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        self.conn = psycopg2.connect(
            # f"host=localhost port=54321 dbname=intex user=adm_intex password=Dd7iKZUP3wYmE7pPvK"
            f"dbname={self.dataset}_reviews port=5432 user=behrooz password=212799"
        )
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, type, value, traceback):
        self.cur.close()
        self.conn.close()

    # Once the query string is determined, this function executes the query over the Postgres database.
    def execute_query(self, query_type, query_string, query_parameters=None):

        # In case the query is relevance retrival, the output should be a list.
        # In other cases, it should be a dictionary.
        results = {}

        self.cur.execute(query_string)
        query_results = self.cur.fetchall()

        # In case the query is rating retrieval, the outcome should be case to integer.
        if query_type in ["get_ratings"]:
            for query_result in query_results:
                results[query_result[0]] = int(query_result[1])

        # The case of key-value retrievals
        elif query_type in ["get_reviews", "get_summaries", "topic_definition"]:
            for query_result in query_results:
                results[query_result[0]] = query_result[1]

        # The case of list retrievals (whose list memebers should be seperated by a space)
        elif query_type in ["get_topics", "get_sentiments"]:
            for query_result in query_results:
                results[query_result[0]] = query_result[1].split(" ")

        # All other normal cases
        else:
            for query_result in query_results:
                results = query_result

        # In an exceptional case, if the executed query yields no result, the function returns False.
        if len(results) == 0:
            return False

        return results

    # Most query executions in INTEX are "select queries". However, there exist a few manipulation
    # ... queries as well, i.e., "insert". These queries are handled by this function.
    def manipulation_query(self, topic_insert_query):

        self.cur.execute(topic_insert_query)
        self.conn.commit()
        return True

    # Each query is identified with a name, i.e., "query_type".
    # If some parameters are required to make the query specific, they are provided in "parameters" as a list.
    def get_query(self, query_type, parameters=None):

        query = ""

        # The query to retrieve elements relevant to an input element (in "parameters[0]"), order defined in "parameters[1]" and limit in "parameters[2]"
        if query_type == "relevance":
            query = "select rev1, rev2 from review_sim where (rev1=" + str(parameters[0]) + " or rev2=" + str(
                parameters[0]) + ") order by "+parameters[1]+" desc limit "+str(parameters[2])+";"

        # The query to retrieve reviews ratings for a given set of elements (indicated in "parameters[0]")
        elif query_type == "get_ratings":
            query = "select id, overall from reviews where in_sample = TRUE and id in (" + \
                parameters[0]+")"

        # The query to retrieve the textual content of reviews for a given set of elements (indicated in "parameters[0]")
        elif query_type == "get_reviews":
            query = "select id, reviewtext from reviews where in_sample = TRUE and id in (" + \
                parameters[0]+")"

        # The query to retrieve summaries for a given set of elements (indicated in "parameters[0]")
        elif query_type == "get_summaries":
            query = "select id, summary from reviews where in_sample = TRUE and id in (" + \
                parameters[0]+")"

        # The query to get item (product) information under review (given the review_id in "parameters[0]")
        elif query_type == "product_info":
            query = "select t2.asin, t2.title, t2.category, t2.price from reviews t1, items t2 where t1.id = " + \
                str(parameters[0])+" and t1.asin = t2.asin;"

        # The query to get review information for the review whose id is indicated in "parameters[0]"
        elif query_type == "reviews_info":
            query = "select asin, overall, summary, reviewtext from reviews where id = " + \
                str(parameters[0])

        # The query to get the sentiment values for the review whose id is indicated in "parameters[0]"
        elif query_type == "sentiment":
            query = "select rev_sentiment from sentiment where rid = " + \
                str(parameters[0])

        # The query to get the topic values for the review whose id is indicated in "parameters[0]"
        elif query_type == "topic":
            query = "select topics from topic where rid = "+str(parameters[0])

        # The query to get the topic values of ALL reviews.
        elif query_type == "get_topics":
            query = "select rid, topics from topic;"

        # The query to get the sentiment values of ALL reviews.
        elif query_type == "get_sentiments":
            query = "select rid, rev_sentiment from sentiment;"

        elif query_type == "topic_insert":
            query = "insert into topic (rid, topics) values (" + \
                str(parameters[0])+", '"+parameters[1]+"');"

        elif query_type == "topic_similarity_insert":
            query = "update review_sim set topic_sim = " + \
                str(parameters[0])+" where rev1 = "+str(parameters[1]
                                                        )+" and rev2 = "+str(parameters[2])+";"
        elif query_type == "tag_insert":
            query = "insert into tag (rid, tags) values (" + \
                str(parameters[0])+", '"+parameters[1]+"');"
        elif query_type == "sentiment_insert":
            query = "insert into sentiment (rid, rev_sentiment) values ("+str(
                parameters[0])+", '"+parameters[1]+"');"

        elif query_type == "sentiment_similarity_insert":
            query = "update review_sim set sentiment = " + \
                str(parameters[0])+" where rev1 = "+str(parameters[1]
                                                        )+" and rev2 = "+str(parameters[2])+";"
        elif query_type == "text_similarity_insert" and parameters[3] == "summary":
            query = "update review_sim set summary_sim = " + \
                str(parameters[0]) + "where rev1 = " + str(parameters[1]
                                                           ) + " and rev2 = " + str(parameters[2]) + ";"
        elif query_type == "text_similarity_insert" and parameters[3] == "review":
            query = "insert into review_sim (rev1,rev2,sim) values ("+str(
                parameters[1])+","+str(parameters[2])+","+str(parameters[0])+");"

        elif query_type == "previous_sample_remove":
            query = "update reviews set in_sample = FALSE;"
        elif query_type == "sample_mark":
            query = "update reviews set in_sample = TRUE where id = " + \
                str(parameters[0])+";"
        elif query_type == "keywords":
            query = "select id from elements as e inner join attributeswhere reviewtext like '" + \
                str(parameters[0])+"' and in_sample = TRUE limit 1;"
        elif query_type == "random_review":
            query = "select id from reviews where in_sample = TRUE order by random() limit 1;"
        elif query_type == "session_existed":
            query = "select iterations_so_far from session where name = '" + \
                parameters[0]+"';"
        elif query_type == "iteration_update":
            query = "update session set iterations_so_far = " + \
                str(parameters[0])+" where name = '"+parameters[1]+"'"
        elif query_type == "session_create":
            query = f"""insert into session (name, iterations_so_far, target) values 
            ('{parameters[0]}',1,'{parameters[1]}');"""
        elif query_type == "text_score":
            query = "select sim from review_sim where rev1 = " + \
                parameters[0]+" and rev2 = "+parameters[1]+";"
        elif query_type == "summary_score":
            query = "select summary_sim from review_sim where rev1 = " + \
                parameters[0]+" and rev2 = "+parameters[1]+";"
        elif query_type == "topic_score":
            query = "select topic_sim from review_sim where rev1 = " + \
                parameters[0]+" and rev2 = "+parameters[1]+";"
        elif query_type == "sentiment_score":
            query = "select sentiment_sim from review_sim where rev1 = " + \
                parameters[0]+" and rev2 = "+parameters[1]+";"
        elif query_type == "tag_score":
            query = "select tag_sim from review_sim where rev1 = " + \
                parameters[0]+" and rev2 = "+parameters[1]+";"
        elif query_type == "target_retrieve":
            query = "select target from session where name = '" + \
                parameters[0]+"';"
        elif query_type == "topic_definition_insert":
            query = "insert into topic_definition (topic_id, words) values ("+str(
                parameters[0])+", '"+str(parameters[1])+"')"
        elif query_type == "topic_definition_truncate":
            query = "truncate table topic_definition;"
        elif query_type == "topic_truncate":
            query = "truncate table topic;"
        elif query_type == "topic_definition":
            query = "select topic_id, words from topic_definition;"
        elif query_type == "iteration_register":
            query = f"""insert into exploration_iteration 
            (session_name, iteration, feedback, guidance, terminate) 
            values ('{parameters[0]}',{parameters[1]},{parameters[2]},ARRAY{parameters[3]},{parameters[4]});"""
        elif query_type == "get_tag":
            query = "select tags from tag where rid = "+str(parameters[0])

        return query

    def get_item_id_from_keywords(self, keywords):
        self.cur.execute(
            f"""select e.id 
                from elements as e 
                inner join attributes as a on e.id = a.element_id and a.name = 'text' 
                where e.type = 'item' and a.value like ' {keywords}' 
                limit 1;""")
        result = self.cur.fetchone()
        return result[0] if result != None else None

    def get_item_count(self):
        self.cur.execute(
            f"""select count(e.id) 
                from elements as e 
                where e.type = 'item';""")
        return self.cur.fetchone()[0]

    def get_random_element_id(self, element_type=None):
        self.cur.execute(
            f"""select e.id 
                from elements as e 
                order by random() limit 1;""")
        result = self.cur.fetchone()
        return result[0] if result != None else None

    def get_random_element(self, element_type=None):
        return pd.read_sql(
            f"""select e.*
                from elements as e 
                order by random() limit 1;""", self.conn).iloc[0]

    def get_element(self, element_id):
        return pd.read_sql(
            f"""select e.*  
                from elements as e 
                where e.id = {element_id};""", self.conn).iloc[0]

    def get_elements(self, element_ids):
        element_ids = str(element_ids).replace(
            '[', '(').replace(']', ')')
        return pd.read_sql(
            f"""select e.*  
                from elements as e 
                where e.id in {element_ids};""", self.conn)

    def get_element_pair_similarity(self, element1_id, element2_id):
        return pd.read_sql(
            f"""select *  
                from similarities
                where (element1_id = {element1_id} and element2_id = {element2_id})
                    or (element1_id = {element2_id} and element2_id = {element1_id});""", self.conn)

    def get_element_to_element_list_similarities(self, exploration_element_ids, element_ids):
        element_ids = str(element_ids).replace(
            '[', '(').replace(']', ')')
        exploration_element_ids = str(exploration_element_ids).replace(
            '[', '(').replace(']', ')')
        return pd.read_sql(
            f"""select *  
                from similarities
                where (element1_id in {exploration_element_ids} and element2_id in {element_ids})
                    or (element1_id in {element_ids} and element2_id in {exploration_element_ids});""", self.conn)

    def get_relevant_elements(self, element, relevance_function, k_prime):
        query = f"""select e.*, s.{relevance_function}  
                from similarities as s
                inner join elements as e on e.id != {element.id} and (e.id = s.element1_id or e.id = s.element2_id) 
                where ( s.element1_id = {element.id}  or s.element2_id = {element.id} ) """
                # and e.type = '{element_type}' """

        if element.type == "item":
            query += f"and (e.type = 'item' or e.item_id = {element.id})"

        query += f"""
                order by s.{relevance_function} desc, e.id 
                limit {k_prime};"""
        return pd.read_sql(query, self.conn)

    def get_episode_start_elements(self, k):
        order_by = "id"
        if configuration.environment_configurations["start_random"]:
            order_by = "random()"
        return pd.read_sql(f"""SELECT * FROM elements order by {order_by} limit {k+1}; """, self.conn)

    def get_related_items(self, related_item_ids):
        if len(related_item_ids) == 0:
            return pd.DataFrame()
        else:
            related_item_ids = str(related_item_ids).replace(
                '[', '(').replace(']', ')')
            return pd.read_sql(
                f"""select e.* 
                from elements as e
                where e.id in {related_item_ids};""", self.conn)

    def get_max_text_size(self):
        self.cur.execute(
            f"select max(length(text)) from elements;")
        return self.cur.fetchone()[0]

    def get_max_tag_count(self):
        self.cur.execute(
            f"select max(array_length(tags,1)) from elements;")
        return self.cur.fetchone()[0]

    def get_target_item_ids(self, query): # core items to select for the experiments
        target_item_ids = pd.read_sql(query, self.conn).item_id.to_list()
        return target_item_ids

    def get_target_ids(self, query):
        target_ids = pd.read_sql(query, self.conn).id.to_list()
        if configuration.learning_configurations["target_limit"]:
            limit = configuration.learning_configurations["target_size"]
            return target_ids[:limit]
        return target_ids
