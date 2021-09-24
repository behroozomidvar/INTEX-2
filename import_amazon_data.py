import random
import pandas as pd
from data.db_interface import DBInterface
from data.generate_review_metadata import Sentiment, Topic, Tag, Similarity
from tqdm import tqdm
import json

MIN_REVIEW_PER_ITEM = 400
REVIEW_SAMPLE_PER_ITEM = 50
SAMPLE_SIZE = 100
with DBInterface("amazon") as db_interface:
    db_interface.cur.execute(
        "DELETE FROM attributes;")
    db_interface.cur.execute(
        "DELETE FROM similarities;")
    db_interface.cur.execute(
        "DELETE FROM elements;")
    db_interface.cur.execute(
        "DELETE FROM exploration_iteration;")
    db_interface.cur.execute(
        "DELETE FROM session;")
    db_interface.conn.commit()

    # Restricted the items to those with price and at least 10 reviews

    item_ids = pd.read_sql(f"""select id
        from (
        select i.id, i.asin, count(r.id) as review_count
        from items as i
        inner join reviews as r on r.asin = i.asin and i.price != ''
        group by i.id, i.asin ) as c
        where review_count >= {MIN_REVIEW_PER_ITEM}  """, db_interface.conn).sample(SAMPLE_SIZE)["id"].to_list()
    items = pd.read_sql(
        f""" select * from items where id in ({','.join(map(str,item_ids))}) """, db_interface.conn)
    element_texts = []
    print("Selecting items and reviews")
    for index, item in tqdm(items.iterrows()):
        reviews = pd.read_sql(
            f"select * from reviews where asin = '{item.asin}'", db_interface.conn).sample(REVIEW_SAMPLE_PER_ITEM)
        item_rating = reviews.overall.sum()/len(reviews)
        item_text = item.description.replace("'", "''")
        item_summary = ""
        attributes_json = {
            'title': item.title.replace("'", "''"),
            'price': item.price,
            "categories": list(map(lambda x: x.replace("'", "''"), item.category)),
            "main_cat": item.main_cat,
        }
        db_interface.cur.execute(
            f"INSERT INTO elements (sentiments, tags, topics, type, text, summary, rating, attributes) VALUES (NULL, NULL, NULL, 'item', '{item_text}', '{item_summary}', {item_rating}, '{json.dumps(attributes_json)}') RETURNING id;")
        item_element_id = db_interface.cur.fetchone()[0]
        db_interface.cur.execute(f"""INSERT INTO public.attributes(name, value, element_id) VALUES 
        ('title', '{item.title.replace("'", "''")}' , {item_element_id}),
        ('price', '{item.price}' , {item_element_id}),
        ('category', '{str(item.category).replace("'", "''")}', {item_element_id}),
        ('main_cat', '{item.main_cat}' , {item_element_id});""")
        db_interface.conn.commit()
        element_texts.append((item_element_id, item.description))

        for index, review in reviews.iterrows():
            review_text = review.reviewtext.replace("'", "''")
            review_summary = review.summary.replace("'", "''")
            attributes_json = {
                'helpful_true': review.helpful_true,
                'reviewtime': item.price,
                "helpful_all": list(map(lambda x: x.replace("'", "''"), item.category))
            }
            if "reviewername" in review and review.reviewername != None:
                attributes_json["reviewername"] = review.reviewername.replace(
                    "'", "''")
            db_interface.cur.execute(
                f"INSERT INTO elements (sentiments, tags, topics, type, item_id, text, summary, rating, attributes) VALUES (NULL, NULL, NULL, 'review', {item_element_id}, '{review_text}', '{review_summary}', {review.overall}, '{json.dumps(attributes_json)}') RETURNING id;")
            review_element_id = db_interface.cur.fetchone()[0]
            insert_query = f"""INSERT INTO public.attributes(name, value, element_id) VALUES 
                    ('helpful_true', '{review.helpful_true}' , {review_element_id}),
                    ('reviewtime', '{review.reviewtime}', {review_element_id}),
                    ('helpful_all', '{review.helpful_all}', {review_element_id})
                    """

            if "reviewername" in review and review.reviewername != None:
                insert_query += f""",('reviewername', '{review.reviewername.replace("'", "''")}', {review_element_id});"""
            else:
                insert_query += ";"
            db_interface.cur.execute(insert_query)
            db_interface.conn.commit()
            element_texts.append((review_element_id, review.reviewtext))

    Sentiment(element_texts, db_interface).generate_sentiments()
    Topic(element_texts, db_interface).generate_topics()
    Tag(element_texts, db_interface).generate_tags()

    Similarity('model/glove.6B.50d.txt',
               db_interface).generate_summary_review_relevance()
