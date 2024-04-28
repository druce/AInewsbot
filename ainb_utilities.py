import logging
import os
import unicodedata

import numpy as np
import pandas as pd
import sqlite3

from scipy.spatial.distance import cdist

from ainb_const import SQLITE_DB

############################################################################################################
# utility functions
############################################################################################################

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AInewsbot")


def log(action_str, source_str="", level=logging.INFO):
    """Output a log message with timestamp, action, source, and severity level.

    Args:
        action_str (str): message
        source_str (str, optional): service or function that generated the message
        level (int, optional): 0: INFO; 1: WARNING; 2: ERROR;. Defaults to 0.

    Returns:
        int: severity
    """
    if source_str:
        message = f"{source_str} - {action_str}"
    else:
        message = action_str

    logger.log(level, message)

    return level


# Example usage
# log("This is a test message", "TestModule", logging.INFO)

def delete_files(download_dir):
    """
    Deletes non-hidden files in the specified directory.

    Args:
        download_dir (str): The path to the directory containing the files.

    Raises:
        OSError: If there is an error while deleting a file.

    Note:
        This function does not delete subdirectories. If you want to remove subdirectories as well, use os.rmdir().

    Example:
        >>> delete_files('/path/to/directory')
    """

    # Iterate over all files in the directory
    for filename in os.listdir(download_dir):
        if filename.startswith('.'):
            continue
        file_path = os.path.join(download_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                # If you want to remove subdirectories as well, use os.rmdir() here
                pass
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def insert_article(conn, cursor, src, title, url, isAI, article_date):
    """
    Inserts a new article into the database.

    Args:
        conn (sqlite3.Connection): The connection object to the SQLite database.
        cursor (sqlite3.Cursor): The cursor object to execute SQL statements.
        src (str): The source of the article.
        title (str): The title of the article.
        url (str): The URL of the article.
        isAI (bool): Indicates whether the article is related to AI.
        article_date (str): The date of the article.

    Raises:
        sqlite3.IntegrityError: If there is a duplicate entry for the URL.
        Exception: If any other error occurs during the insertion process.

    Returns:
        None
    """
    try:
        cursor.execute("INSERT OR IGNORE INTO news_articles (src, title, url, isAI, article_date) VALUES (?, ?, ?, ?, ?)",
                       (src, title, url, isAI, article_date))
        conn.commit()
    except sqlite3.IntegrityError:
        log(f"Duplicate entry for URL: {url}")
    except Exception as err:
        log(err)


def filter_unseen_urls_db(orig_df):
    """
    Filters out rows from orig_df that have URLs already present in the database.

    Args:
        orig_df (pandas.DataFrame): The original DataFrame containing the URLs.

    Returns:
        pandas.DataFrame: The filtered DataFrame with rows removed if their URLs are already present in the database.
    """
    conn = sqlite3.connect(SQLITE_DB)
    existing_urls = pd.read_sql_query("SELECT url FROM news_articles", conn)
    conn.close()

    existing_urls_list = existing_urls['url'].tolist()
    log(f"Existing URLs: {len(existing_urls_list)}")

    filtered_df = orig_df[~orig_df['url'].isin(existing_urls_list)]
    log(f"New URLs: {len(filtered_df)}")
    return filtered_df


def unicode_to_ascii(input_string):
    """
    Converts a Unicode string to ASCII by normalizing it to NFKD form and then encoding it to ASCII bytes.
    Characters that cannot be converted are ignored.

    Args:
        input_string (str): The Unicode string to be converted.

    Returns:
        str: The converted ASCII string.

    """
    normalized_string = unicodedata.normalize('NFKD', input_string)
    ascii_bytes = normalized_string.encode('ascii', 'ignore')
    ascii_string = ascii_bytes.decode('ascii')
    return ascii_string


def nearest_neighbor_sort(embedding_array, start_index=0):
    """
    Sorts the embeddings in a greedy traveling salesman traversal order based on their pairwise Euclidean distances.

    Args:
        embedding_array (ndarray): The array of embeddings.
        start_index (int, optional): The index of the starting embedding. Defaults to 0.

    Returns:
        ndarray: The sorted array of indices representing the order of embeddings in the traversal.

    Examples:
        >>> embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> nearest_neighbor_sort(embeddings)
        array([0, 1, 2])
    """
    # Compute the pairwise Euclidean distances between all embeddings
    distances = cdist(embedding_array, embedding_array, metric='euclidean')

    # Start from the first headline as the initial point
    path = [start_index]
    visited = set(path)

    while len(path) < len(embedding_array):
        last = path[-1]
        # Set the distances to already visited nodes to infinity to avoid revisiting
        distances[:, last][list(visited)] = np.inf
        # Find the nearest neighbor
        nearest = np.argmin(distances[:, last])
        path.append(nearest)
        visited.add(nearest)

    return np.array(path)
