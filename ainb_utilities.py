import logging
import os
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import sqlite3

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import linear_sum_assignment

from ainb_const import SQLITE_DB

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging
import os
import unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from scipy.spatial.distance import cdist, pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import linear_sum_assignment
from ainb_const import SQLITE_DB
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

############################################################################################################
# utility functions
############################################################################################################
VERBOSE = 1
# Configure logging
# captures a ton of info from selenium and bs4 if level is DEBUG, might crash Jupyter

# Check if VERBOSE variable exists and is set to 1
if 'VERBOSE' in globals() and VERBOSE == 1:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("AInewsbot")
else:
    logger = None


def log(action_str, source_str="", level=logging.INFO):
    """Output a log message with timestamp, action, source, and severity level.

    Args:
        action_str (str): message
        source_str (str, optional): service or function that generated the message
        level (int, optional): 0: INFO; 1: WARNING; 2: ERROR;. Defaults to 0.

    Returns:
        int: severity
    """
    if logger:
        if source_str:
            message = f"{str(source_str)} - {str(action_str)}"
        else:
            message = str(action_str)

        logger.log(level, message)

    return None


# Example usage
# log("This is a test message", "TestModule", logging.INFO)
# Configure logging
# captures a ton of info from selenium and bs4 if level is DEBUG, might crash Jupyter
############################################################################################################
# utility functions
############################################################################################################


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


def insert_article(conn, cursor, src, actual_src, title, url, actual_url, isAI, article_date):
    """
    Inserts a new article record into the SQLite database.

    Args:
        conn (sqlite3.Connection): The connection object to the SQLite database.
        cursor (sqlite3.Cursor): The cursor object to execute SQL statements.
        src (str): The source of the article.
        title (str): The title of the article.
        url (str): The URL of the article.
        actual_url (str): The actual URL of the article (URL shorteners and aggregators like Google News will redirect url to actual_url).
        isAI (bool): Indicates whether the article is related to AI.
        article_date (str): The date of the article.

    Raises:
        sqlite3.IntegrityError: If there is a duplicate entry for the URL.
        Exception: If any other error occurs during the insertion process.

    Returns:
        None
    """
    try:
        cursor.execute("INSERT OR IGNORE INTO news_articles (src, actual_src, title, url, actual_url, isAI, article_date, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (src, actual_src, title, url, actual_url, isAI, article_date, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
    except sqlite3.IntegrityError:
        log(f"Duplicate entry for URL: {url}")
    except Exception as err:
        log(err)


def filter_unseen_urls_db(orig_df, before_date=None, after_date=None):
    """
    Filters out rows from orig_df that have URLs already present in the database during the specified time interval.

    Args:
        orig_df (pandas.DataFrame): The original DataFrame containing the URLs.
        before_date (str): only query database for URLs from before this date
        (e.g. rerun on todays urls even if already in database by specifying before_date = today)
        after_date (str): only query database after this date
        (e.g. as database gets large, only check against urls in last year by specifying after_date = 1 year ago)

    Returns:
        pandas.DataFrame: The filtered DataFrame with rows removed if their URLs are already present in the database.
    """
    conn = sqlite3.connect(SQLITE_DB)
    where_clause = ''
    if before_date:
        where_clause = f"WHERE timestamp < '{before_date}'"
    if after_date:
        if len(where_clause) == 0:
            where_clause = f"WHERE timestamp > '{after_date}'"
        else:
            where_clause += f" AND timestamp > '{after_date}'"
    log(f"Querying SQLite with where_clause: {where_clause}")
    existing_urls = pd.read_sql_query(
        f"SELECT url, src, title FROM news_articles {where_clause}", conn)
    conn.close()

    log(f"URLs in orig_df: {len(orig_df)}")

    existing_urls_list = existing_urls['url'].tolist()
    log(f"Existing URLs in DB: {len(existing_urls_list)}")
    filtered_df = orig_df[~orig_df['url'].isin(existing_urls_list)]
    log(f"New URLs in df filtered by URL: {len(filtered_df)}")

    existing_urls = existing_urls.drop('url', axis=1)
    existing_urls = existing_urls.drop_duplicates()
    drop_df = pd.merge(existing_urls, filtered_df,
                       how='inner',
                       on=['src', 'title'])[["src", "title", "id"]]
    log(f"Existing src+title: {len(drop_df)}")

    filtered_df = filtered_df.loc[~filtered_df["id"].isin(drop_df["id"])]
    log(f"New URLs in df filtered by src+title: {len(filtered_df)}")

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
    # Normalize the Unicode string to NFKD form
    normalized_string = unicodedata.normalize('NFKD', input_string)

    # Encode to ASCII bytes, ignoring characters that cannot be converted
    ascii_bytes = normalized_string.encode('ascii', 'ignore')

    # Convert bytes back to a string
    ascii_string = ascii_bytes.decode('ascii')

    return ascii_string


def nearest_neighbor_sort(embedding_array, start_index=None):
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

    if start_index is None:
        # Find closest neighbor to the centroid
        centroid = np.mean(embedding_array, axis=0)
        centroid_dist = np.linalg.norm(embedding_array - centroid, axis=1)
        start_index = np.argmin(centroid_dist)

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


def agglomerative_cluster_sort(embedding_df):
    """
    Sorts embeddings using agglomerative clustering. Neither sort works perfectly, unclear which is better.
    Agglomerative clustering divides the data into 2 clusters minimizing in-cluster distances and maximizing
    between-cluster distances. Then you can recursively repeat on each cluster to get a hierarchy of clusters.
    If you do a breadth-first traversal of the leaves which contain a single element, you get a sort order
    that heuristically approximately minimizes the sum of distances between adjacent elements.

    Parameters:
    embedding_df (pandas.DataFrame): The dataframe containing the embeddings of the topics.

    Returns:
    list: An nparray of the sort order of the dataframe rows based on the agglomerative clustering.

    """
    distance_matrix = pdist(embedding_df.values, metric='cosine')
    Z = linkage(distance_matrix, method='ward')
    leaf_order = leaves_list(Z)
    return leaf_order


def traveling_salesman_sort_scipy(df):
    """
    Given a dataframe of embeddings, sort the rows of the dataframe based on the order of nodes in the traveling salesman traversal.

    Args:
        df (pandas.DataFrame): The input dataframe to be sorted.

    Returns:
        numpy.ndarray: The sorted dataframe.

    """

    # Convert the dataframe to a distance matrix
    distance_matrix = cdist(df, df, metric='euclidean')

    # Solve the linear sum assignment problem to find the optimal assignment of rows
    row_indices, _ = linear_sum_assignment(distance_matrix)

    return row_indices


def traveling_salesman_sort_ortools(df):
    """
    Given a dataframe of embeddings, sort the rows of the dataframe based on the order of nodes in the traveling salesman traversal
    which minimizes the euclidean distance traveled in embedding space.

    Args:
        df (pandas.DataFrame): The input dataframe to be sorted.

    Returns:
        numpy.ndarray: The sort order.

    """
    # Convert the dataframe to a distance matrix
    distance_matrix = cdist(df, df, metric='euclidean')

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)

    # Create the routing model
    routing = pywrapcp.RoutingModel(manager)

    # Create the distance callback
    def distance_callback(from_index, to_index):
        return distance_matrix[from_index][to_index]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define the cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set the search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Get the order of the nodes in the solution
    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    return order


def send_gmail(subject, html_str):
    # body
    body = f"""
    <html>
        <head></head>
        <body>
        <div>
        {html_str}
        </div>
        </body>
    </html>
    """
    # Setup the MIME
    message = MIMEMultipart()
    email_address = os.getenv("GMAIL_USER")
    message['From'] = email_address
    message['To'] = email_address
    message['Subject'] = subject
    message.attach(MIMEText(body, 'html'))

    # Create SMTP session
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()  # Secure the connection
        server.login(email_address, os.getenv("GMAIL_PASSWORD"))
        text = message.as_string()
        server.sendmail(email_address, email_address, text)
