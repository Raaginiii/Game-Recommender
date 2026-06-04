import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize


MODEL_PATH = Path(__file__).with_name("model.pkl")


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def clean_feature_name(value: str, prefix: str) -> str:
    value = (
        value.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("&", "and")
    )
    return f"{prefix}_{value}"


def readable_options(prefix: str, feature_columns: list[str]) -> list[str]:
    options = []
    prefix_text = f"{prefix}_"
    for column in feature_columns:
        if column.startswith(prefix_text):
            options.append(column.removeprefix(prefix_text).replace("_", " "))
    return sorted(options)


def recommend_by_name(game_name: str, count: int, knn_model, x_normalized, df):
    matches = df[df["name"].str.contains(game_name, case=False, na=False)]
    if matches.empty:
        return None, pd.DataFrame()

    game_index = matches.index[0]
    selected_game = matches.iloc[0]["name"]
    neighbors = min(count + 1, len(df))
    distances, indices = knn_model.kneighbors(
        x_normalized[game_index].reshape(1, -1),
        n_neighbors=neighbors,
    )

    rows = []
    for distance, index in zip(distances[0][1:], indices[0][1:]):
        rows.append(
            {
                "Game": df.iloc[index]["name"],
                "Similarity": f"{round((1 - distance) * 100, 2)}%",
            }
        )

    return selected_game, pd.DataFrame(rows)


def recommend_by_features(
    genres: list[str],
    platforms: list[str],
    tags: list[str],
    count: int,
    knn_model,
    feature_columns: list[str],
    df,
):
    custom_features = np.zeros(len(feature_columns))

    selected_columns = []
    selected_columns.extend(clean_feature_name(value, "genre") for value in genres)
    selected_columns.extend(clean_feature_name(value, "platform") for value in platforms)
    selected_columns.extend(clean_feature_name(value, "tag") for value in tags)

    for column in selected_columns:
        if column in feature_columns:
            custom_features[feature_columns.index(column)] = 1

    if not custom_features.any():
        return pd.DataFrame()

    custom_features = normalize(custom_features.reshape(1, -1), norm="l2")
    neighbors = min(count, len(df))
    distances, indices = knn_model.kneighbors(custom_features, n_neighbors=neighbors)

    rows = []
    for distance, index in zip(distances[0], indices[0]):
        rows.append(
            {
                "Game": df.iloc[index]["name"],
                "Similarity": f"{round((1 - distance) * 100, 2)}%",
            }
        )

    return pd.DataFrame(rows)


st.set_page_config(page_title="Game Recommender", layout="centered")

st.title("Game Recommender")
st.write("Choose a game you like, or select a few preferences to get recommendations.")

try:
    knn_model, x_normalized, feature_columns, df = load_model()
except Exception as exc:
    st.error(f"Could not load model.pkl: {exc}")
    st.stop()

if "name" not in df.columns:
    st.error("The model data must include a 'name' column.")
    st.stop()

feature_columns = list(feature_columns)
game_names = sorted(df["name"].dropna().unique())

tab_game, tab_features = st.tabs(["By game", "By preferences"])

with tab_game:
    game_name = st.selectbox("Game", game_names)
    count = st.slider("Number of recommendations", 1, 20, 6, key="name_count")

    if st.button("Recommend similar games", type="primary"):
        selected_game, recommendations = recommend_by_name(
            game_name,
            count,
            knn_model,
            x_normalized,
            df,
        )

        if recommendations.empty:
            st.warning("No matching game found.")
        else:
            st.subheader(f"Because you liked {selected_game}")
            st.dataframe(recommendations, hide_index=True, use_container_width=True)

with tab_features:
    genres = st.multiselect("Genres", readable_options("genre", feature_columns))
    platforms = st.multiselect("Platforms", readable_options("platform", feature_columns))
    tags = st.multiselect("Tags", readable_options("tag", feature_columns))
    count = st.slider("Number of recommendations", 1, 20, 6, key="feature_count")

    if st.button("Recommend from preferences", type="primary"):
        recommendations = recommend_by_features(
            genres,
            platforms,
            tags,
            count,
            knn_model,
            feature_columns,
            df,
        )

        if recommendations.empty:
            st.warning("Select at least one genre, platform, or tag.")
        else:
            st.subheader("Recommended games")
            st.dataframe(recommendations, hide_index=True, use_container_width=True)
