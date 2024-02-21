import duckdb
import os
from glob import glob


from subreddits import (
    location,
    gender_females,
    gender_males,
    age_groups,
    occupations,
    pobp,
    married,
    income,
    education_level,
)

all_keys = (
    location
    + gender_females
    + gender_males
    + list(age_groups.keys())
    + list(education_level.keys())
    + occupations
    + pobp
    + married
    + income
)
formatted_strings = ",".join(["'%s'" % item for item in all_keys])


con = duckdb.connect("reddit_comments_database.duckdb")
cur = con.cursor()
cur.execute("SET enable_progress_bar=true")

# Creating the Reddit table
cur.execute(
    """
CREATE TABLE IF NOT EXISTS comments (
    created_utc VARCHAR,
    controversiality INT,
    body VARCHAR,
    subreddit_id VARCHAR,
    id VARCHAR,
    score INT,
    author VARCHAR,
    subreddit VARCHAR,
    link_id VARCHAR,
)
"""
)


folder = "/home/robin/Downloads/reddit"

# List all files in the folder


def insert_into_duckdb(file):
    insert_query = (
        f"INSERT INTO comments SELECT created_utc, controversiality, body, subreddit_id, id, score, author, subreddit, link_id "
        f"FROM read_parquet('{file}') WHERE author != '[deleted]' AND length(body) >=10 "
        f"AND subreddit IN ({formatted_strings}) ORDER BY author"
    )

    cur.execute(insert_query)


# insert data into DuckDB
con.execute("BEGIN TRANSACTION")
try:
    files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], "*.parquet"))]
    for file in files:
        # if "2012" in file or "2013" in file or "2014" in file:
        print(f"Inserting {file}")
        insert_into_duckdb(file)
        print(f"Inserted {file}")
    con.execute("COMMIT TRANSACTION")

    con.execute("BEGIN TRANSACTION")
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS author_aggregated AS SELECT author, COUNT(*), array_agg(body), array_agg(subreddit), array_agg(created_utc) FROM comments GROUP BY author"
    )
    cur.execute("DROP TABLE comments")
    con.execute("COMMIT TRANSACTION")
except Exception as e:
    print(e)
    con.execute("ROLLBACK TRANSACTION")
finally:
    cur.close()
