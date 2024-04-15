%matplotlib inline
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD
# from surprise.model_selection import cross_validate  # for the evaluate 

# import warnings; warnings.simplefilter('ignore')

movie_data = pd.read_csv('movies_metadata.csv')

# '''md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])'''
# movie_data['genres'] = movie_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# movie_data['year'] = pd.to_datetime(movie_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# vote_counts = movie_data[movie_data['vote_count'].notnull()]['vote_count'].astype('int')
# vote_averages = movie_data[movie_data['vote_average'].notnull()]['vote_average'].astype('int')
# mean_vote_averages = vote_averages.mean()
# md = movie_data

# qualified = md[(md['vote_count'] >= minimum_votes) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
# qualified['vote_count'] = qualified['vote_count'].astype('int')
# qualified['vote_average'] = qualified['vote_average'].astype('int')


# small_mdf = pd.read_csv('links_small.csv')
# small_mdf = small_mdf[small_mdf['tmdbId'].notnull()]['tmdbId'].astype('int')

# def convert_int(x):
#     try:
#         return int(x)
#     except:
#         return np.nan


# md['id'] = md['id'].apply(convert_int)
# md[md['id'].isnull()]


# md = md.drop([19730, 29503, 35587])
# md['id'] = md['id'].astype('int')
# sm_df = md[md['id'].isin(small_mdf)]


# sm_df['tagline'] = sm_df['tagline'].fillna('')
# sm_df['description'] = sm_df['overview'] + sm_df['tagline']
# sm_df['description'] = sm_df['description'].fillna('')


# tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(sm_df['description'])


# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# sm_df = sm_df.reset_index()
# titles = sm_df['title']
# indices = pd.Series(sm_df.index, index=sm_df['title'])


# def get_recommendations(title):
#     idx = indices[title]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:51]
#     movie_indices = [i[0] for i in sim_scores]
#     return titles.iloc[movie_indices]


# inpt = input()
# try:
#     print(get_recommendations(inpt).head(10))
# except:
#     print("Movie not in the list")
data_temp = movie_data["genres"]

'''md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])'''
movie_data['genres'] = movie_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movie_data['genres'].head(2)
data_temp[:2]
movie_data['release_date']
movie_data['year'] = pd.to_datetime(movie_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


movie_data.columns
len(movie_data.columns)
'''Weighted Rating (WR) =  (v/(v+m).R)+(m/(v+m).C)'''

vote_counts = movie_data[movie_data['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movie_data[movie_data['vote_average'].notnull()]['vote_average'].astype('int')
mean_vote_averages = vote_averages.mean()
mean_vote_averages
minimum_votes = vote_counts.quantile(0.95)
minimum_votes
md=movie_data
qualified = md[(md['vote_count'] >= minimum_votes) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified
qualified["weighted_rating"] = qualified.apply(Weighted_matrix,axis=1)
qualified.shape
qualified = qualified.sort_values("weighted_rating",ascending=False)
qualified.head(10)
temp = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
temp.name = 'genre'
mgen_df = md.drop('genres', axis=1).join(temp)


def make_toplist(genre, percentile=0.85):
    df = mgen_df[mgen_df['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    col_list = ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genre']
    qualified = df[(df['vote_count'] >= m)
                   & (df['vote_count'].notnull())
                   & (df['vote_average'].notnull())][col_list]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['weighted_rating'] = qualified.apply(lambda x:
                                                   (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (
                                                               m / (m + x['vote_count']) * C),
                                                   axis=1)
    qualified = qualified.sort_values('weighted_rating', ascending=False).head(250)

    return qualified
make_toplist('Romance').head(10)
movie_data.columns
movie_data['adult'].head(10)
movie_data[movie_data['adult']=='True']
small_mdf = pd.read_csv('links_small.csv')
small_mdf = small_mdf[small_mdf['tmdbId'].notnull()]['tmdbId'].astype('int')


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


md['id'] = md['id'].apply(convert_int)
md[md['id'].isnull()]

md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
sm_df = md[md['id'].isin(small_mdf)]

sm_df['tagline'] = sm_df['tagline'].fillna('')
sm_df['description'] = sm_df['overview'] + sm_df['tagline']
sm_df['description'] = sm_df['description'].fillna('')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(sm_df['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

sm_df = sm_df.reset_index()
titles = sm_df['title']
indices = pd.Series(sm_df.index, index=sm_df['title'])


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


inpt = input()
try:
    print(get_recommendations(inpt).head(10))
except:
    print("Movie not in the list")

