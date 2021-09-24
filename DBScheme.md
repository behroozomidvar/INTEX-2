# The review database

## The database schema

The reviews and related data are first imported from the original dataset tables into the intex data schema 
The amazon database dump contains both the original tables and the intex tables

- **Original tables**. This group contains two tables, `meta` and `reviews`. The table `meta` contains information about products. Important attributes in this table are `id` (incremental unique ID, and the primary key), `asin` (product ID in Amazon database), `title` (product name), `category`, `main_cat` (main category), `description`, `image`. The table `reviews` contains the reviews about the products. Important attributes in this table are `id` (incremental unique ID, and the primary key), `asin` (the product ID of the review), `overall` (rating associated to review, out of 5), `summary`, `reviewtext`, `reviewtime`.

- **Elements table**. Intex treats the items and reviews in the same way and qualifies them as `elements`. The table contains the attributes `type` (item or review), `item_id` with a reference to the element with the related item data for the reviews, `text` and `summary` containing the text data of the elements, `rating` containing the rating for the reviews and the average review ratings for the items, and finally the pre-computed signatures of the elements, `topics` (the LDA topic distribution over 10 topics), `tag` stores auto-generated tags for each review, `sentiment` stores sentiment values for each review, `attributes` stores a json document with the different attributes of the element.

- The **Attributes table** stores a list of unspecified attributes for each elements. Each record has a `name`,a `value`, and a reference to the related `element`. DEPRECATED since the addition of the attributes column to elements. Kept for manual querying ease.

- The **Similarities table**  stores the pre-computed pairwise similarities between `elements`. The table has the following attributes: `element1_id` and `element2_id` (IDs of the first and second elements under comparison, noting that similarity computation is symmetric), `sim` (similarity in the textual content of the elements, a value between 0 and 100), `summary_sim` (similarity between the summary of elements, a value between 0 and 100), `sentiment_sim` (similarity between the sentiment values of the elements, a value between 0 and 1), `tag_sim` (similarity between the tags of the elements, a value between 0 and 100), `topic_sim` (similarity between the topics of the elements, a value between 0 and 1), `attribute_sim` (Jaccard similarity between the sets of attribute/values of the elements)

- **Session management tables**. The tables `session` and `exploration_iteration` store information about the exploration sessions that the user performs in the GUI. The table `session` contains the following attributes: `id` (session ID), `name` (session name), `target` (the target ID to compute reward values in this session), `iterations_so_far` (number of exploration iterations performed so far in this session). The table `exploration_iteration` contains the following attributes: `id` (incremental unique ID, and the primary key), `session_name` (the name of the exploration session that the iteration belongs to), `iteration` (the sequence number of the iteration in the session), `feedback` (the ID of the review selected as the input element in the iteration), `guidance` (the *k* IDs of the elements returned as the output elements in the iteration), `terminate` (a boolean attribute showing whether the iteration is the last one for the session, where the value 1 means it is a terminating session).

- The **topic_definition** table stores top-50 words for each LDA topic.

## (Re)generation of signatures and similarities

To import amazon data, and regenerate all the signatures ans similarities, use the Python script `import_amazon_data.py`. You can edit it to change the SAMPLE_SIZE (the number of items to import with their reviews) at the beginning of the script.

Note that the **GloVe language model is required to run this script**. You can download it [here](https://www.dropbox.com/s/zccfq7sd07dlkem/glove.6B.50d.txt?dl=0) and then put it in a folder called `model`.
