import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
import requests
import json
from data.db_interface import DBInterface
from data.generate_review_metadata import Sentiment, Topic, Tag, Similarity

MIN_REVIEW_PER_ITEM = 400
REVIEW_SAMPLE_PER_ITEM = 50
SAMPLE_SIZE = 1000
PROCESS_COUNT=15
with DBInterface("imdb") as db_interface:
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
    # print("Concatenating reviewId/Movie to count and select the movies")
    # for i in tqdm(range(1, 6)):
    #     raw_review_data = pd.read_json(f"import_data/part-0{i}.json")
    #     if i == 1:
    #         review_movies: pd.DataFrame = raw_review_data[[
    #             "review_id", "movie"]]
    #     else:
    #         review_movies = review_movies.append(
    #             raw_review_data[["review_id", "movie"]])
    
    # review_movies = review_movies.groupby("movie").count()["review_id"].reset_index(
    #     name='review_count').sort_values(['review_count'], ascending=False)
    # review_movies.to_csv("import_data/review_movies.csv", index=False)
    review_movies = pd.read_csv("import_data/review_movies.csv")
    review_movies = review_movies[review_movies.review_count >=
                                  MIN_REVIEW_PER_ITEM]
    review_movie_titles = review_movies.movie.to_list()
    imdb_titles = pd.read_csv("import_data/title_basics.csv")

    imdb_titles = imdb_titles[imdb_titles.primaryTitleWithYear.isin(
        review_movie_titles) & (imdb_titles.titleType == "movie")]

    print(f"{len(imdb_titles)} possible shows to pick")
    selected_movies = imdb_titles.sample(SAMPLE_SIZE)

    print("Concatenating selected movies reviews")
    selected_movies_reviews = DataFrame()
    selected_movie_titles = selected_movies.primaryTitleWithYear.to_list()
    for i in tqdm(range(1, 6)):
        raw_review_data = pd.read_json(f"import_data/part-0{i}.json")
        if len(selected_movies_reviews) == 0:
            selected_movies_reviews: pd.DataFrame = raw_review_data[raw_review_data.movie.isin(
                selected_movie_titles)]
        else:
            selected_movies_reviews = selected_movies_reviews.append(
                raw_review_data[raw_review_data.movie.isin(selected_movie_titles)])
    # selected_movies_reviews.to_csv("import_data/selected_movies_reviews.csv", index=False)
    # selected_movies_reviews = pd.read_csv("import_data/selected_movies_reviews.csv")
    element_texts = []
    print("Importing movies and reviews")
    for index, movie in tqdm(selected_movies.iterrows(), total=selected_movies.shape[0]):
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie.tconst}?api_key=be794be4c63f7d81860f882848e453b8&language=en-US")
        movie_details = response.json()
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie.tconst}/credits?api_key=be794be4c63f7d81860f882848e453b8&language=en-US")
        movie_credits = response.json()
        reviews = selected_movies_reviews[(selected_movies_reviews.movie ==
                                           movie.primaryTitleWithYear) & selected_movies_reviews.rating.notna()].sample(REVIEW_SAMPLE_PER_ITEM)
        if not 'overview' in movie_details:
            print(f"{movie_details} NO OVERVIEW")
            continue
        item_text = movie_details['overview'].replace("'", "''") 
        item_summary = movie_details['tagline'].replace("'", "''")
        director = next(
            (x for x in movie_credits["crew"] if "job" in x and x["job"] == "Director"), None)
        if director == None:
            print(f"{movie_details} {movie_credits} NO DIRECTOR")
            continue
        actors = movie_credits["cast"][0:5]
        item_rating = reviews.rating.mean()
        attributes_json = {
            'title': movie_details["title"].replace("'", "''"),
            'director': director["name"].replace("'", "''"),
            "genres": list(map(lambda x: x["name"].replace("'", "''"), movie_details["genres"])),
            "actors": list(map(lambda x: x["name"].replace("'", "''"), actors)),
            "release": movie.startYear
        }
        db_interface.cur.execute(
            f"INSERT INTO elements (sentiments, tags, topics, type, text, summary, rating, attributes) VALUES (NULL, NULL, NULL, 'item', '{item_text}', '{item_summary}', {item_rating}, '{json.dumps(attributes_json)}') RETURNING id;")
        item_element_id = db_interface.cur.fetchone()[0]
        attributes_query = f"""INSERT INTO public.attributes(name, value, element_id) VALUES 
        ('title', '{movie_details["title"].replace("'", "''")}' , {item_element_id}),
        ('director', '{director["name"].replace("'", "''")}' , {item_element_id}),
        ('release', '{movie.startYear}', {item_element_id}),
        """
        attribute_values = []
        for actor in actors:
            actor_name = actor['name'].replace("'", "''")
            attribute_values.append(
                f"('actor', '{actor_name}' , {item_element_id})")
        for genre in movie_details["genres"]:
            attribute_values.append(
                f"('genre', '{genre['name']}' , {item_element_id})")

        attributes_query += ", ".join(attribute_values)
        db_interface.cur.execute(attributes_query)
        db_interface.conn.commit()
        element_texts.append((item_element_id, movie_details['overview']))

        for index, review in reviews.iterrows():
            review_text = review.review_detail.replace("'", "''")
            review_summary = review.review_summary.replace("'", "''")
            reviewer = review.reviewer.replace("'", "''")
            attributes_json = {
                'helpful': str(review.helpful).replace("'", "''"),
                'reviewtime': review.review_date,
                "reviewername": reviewer
            }
            db_interface.cur.execute(
                f"INSERT INTO elements (sentiments, tags, topics, type, item_id, text, summary, rating, attributes) VALUES (NULL, NULL, NULL, 'review', {item_element_id}, '{review_text}', '{review_summary}', {review.rating}, '{json.dumps(attributes_json)}') RETURNING id;")
            review_element_id = db_interface.cur.fetchone()[0]
            insert_query = f"""INSERT INTO public.attributes(name, value, element_id) VALUES 
                    ('helpful', '{str(review.helpful).replace("'", "''")}' , {review_element_id}),
                    ('reviewtime', '{review.review_date}', {review_element_id}),
                    ('reviewername', '{reviewer}', {review_element_id});
                    """
            db_interface.cur.execute(insert_query)
            db_interface.conn.commit()
            element_texts.append((review_element_id, review_text))

    Sentiment(element_texts, db_interface).generate_sentiments()
    Topic(element_texts, db_interface).generate_topics()
    Tag(element_texts, db_interface).generate_tags()

    Similarity('data/glove.6B.50d.txt',
               db_interface, process_count=PROCESS_COUNT).generate_summary_review_relevance()
