import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

#loading the data files
@st.cache(show_spinner=False)
def read_data():
    team_similarity = pd.read_csv('data/team_similarity.csv')
    team_similarity.set_index('Unnamed: 0', inplace=True)
    stats_df = pd.read_csv('data/player_stats.csv')
    stats_df['player_2'] = stats_df['Player']
    stats_df.set_index('player_2', inplace=True)
    return (team_similarity, stats_df)

team_similarity, stats_df = read_data()

@st.cache(show_spinner=False)
def create_dict(df, items):
    return [dict(zip(df['Player'], df[item])) for item in items]

cols = ['Season', 'Pos', 'Squad', 'Comp', 'Age', 'Foot']
player_mappings = create_dict(stats_df, cols)
player_df = stats_df.copy()
player_df.drop(['Player', 'Pos', 'Squad', 'Comp', '90s', 'Season', 'Squad_2', 'Age', 'Foot'], axis=1, inplace=True)

@st.cache(show_spinner=False)
def scale_features(df1, df2):
    x = StandardScaler().fit_transform(df1)
    X = pd.DataFrame(x, columns=df1.columns, index=df2.index)
    dj = pd.DataFrame(cosine_similarity(X, dense_output=True))
    dict_cols = {ind:entity for ind, entity in enumerate(df2.index.values)}
    dj.rename(dict_cols, axis=1, inplace=True)
    dj.rename(dict_cols, axis=0, inplace=True)
    return dj

player_similarity = scale_features(player_df, stats_df)

st.markdown("<h1 style='text-align: center'>Player Recommendation Tool</h1>", unsafe_allow_html=True)
st.markdown('A player recommendation system which outputs the most similar players for a given player based on various' \
' features collected via Fbref/Statsbomb: https://fbref.com/en/ from 17-18 to 21-22 season for top five European Leagues.')
st.markdown('This type of tool can come in handy in the initial stages of the recruitment process to filter down on the' \
' most similar options for a given player that the club wants to find a replacement for.')
st.markdown("Team similarity is also added as an extra feature to provide more context and  help find" \
" suitable players that not only match individually but can also fit well in the team's particular playing style.")
st.markdown('Cosine similarity is used as a similarity measure which was later normalized in the range of 0 to 100.')

@st.cache
def get_list(df, col):
    items = df[col].unique()
    items = np.insert(items, 0, 'All')
    return items

seasons = get_list(stats_df, 'Season')
leagues = get_list(stats_df, 'Comp')
foot = get_list(stats_df, 'Foot')
positions = get_list(stats_df, 'Pos')
players = stats_df['Player']

select_player = st.sidebar.selectbox(
    'Select Player',
    players)
player_team = player_mappings[2][select_player]

results = st.sidebar.slider('Number Of Results', 5, 20)
min_age = stats_df['Age'].min()
max_age = stats_df['Age'].max()
age_bracket = st.sidebar.slider('Age Bracket', float(min_age), float(max_age), value=[float(min_age), float(max_age)])

select_season = st.sidebar.selectbox(
    'Select Season',
    seasons
)

select_league = st.sidebar.selectbox(
    'Select League',
    leagues
)

select_foot = st.sidebar.selectbox(
    'Select Preferred Foot',
    foot
)

select_pos = st.sidebar.selectbox(
    'Compare With Position: ',
    positions
)

@st.cache(show_spinner=False)
def recommend_players(player, squad, player_data, squad_data, mappings, output, age, season='All', league='All', foot='All', pos='All'):
    temp_df = pd.DataFrame(player_data[player].sort_values(ascending=False)).reset_index()
    temp_df.rename({player:'Player Similarity Score', 'index':'Player'}, axis=1, inplace=True)
    new_cols = ['Season', 'Position', 'Team', 'League', 'Age', 'Preferred Foot']
    for i, col in enumerate(new_cols):
        temp_df[col] = temp_df['Player'].map(mappings[i])
    temp_df['Team Similarity Score'] = temp_df['Team'].apply(lambda x: squad_data.loc[squad, x])
    temp_df['Player Similarity Score'] = normalize(temp_df['Player Similarity Score'])
    temp_df['Team Similarity Score'] = normalize(temp_df['Team Similarity Score'])
    cols = ['Season', 'League', 'Preferred Foot', 'Position']
    params = [season, league, foot, pos]
    filter_dict = dict(zip(cols, params))
    temp_df = filter_df(temp_df, filter_dict)
    temp_df = temp_df[(temp_df['Age'] >= age[0]) & (temp_df['Age'] <= age[1])]
    temp_df = temp_df.iloc[1:output+1, :]
    temp_df.reset_index(inplace=True, drop=True)
    temp_df = temp_df[['Player', 'Player Similarity Score', 'Team Similarity Score', 'Position',
                       'Age', 'Preferred Foot']]

    return temp_df

def normalize(array):
    return np.array([round(num, 2) for num in (array - min(array))*100/(max(array)-min(array))])

def filter_df(df, pairs):
    for key, value in pairs.items():
        if value != 'All':
            df = df[df[key] == value]
    return df

rec_df = recommend_players(select_player, player_team, player_similarity, team_similarity, player_mappings, results,
                           age_bracket, select_season, select_league, select_foot, select_pos)

st.write(f"Showing top {results} players similar to <mark>{select_player}</mark>", unsafe_allow_html=True)
st.table(rec_df)
