from src.team_clustering.utils import extract_embeddings

class ClusteringModel:
    def __init__(self, feature_extraction_model, dimension_reducer, clustering_model, n_clusters=2):
        self.n_clusters = n_clusters
        self.feature_extraction_model = feature_extraction_model
        self.dimension_reducer = dimension_reducer
        self.clustering_model = clustering_model

    def fit_predict(self, images):
        # Extract features
        features = extract_embeddings(self.feature_extraction_model, images)

        # Reduce dimensions
        reduced_features = self.dimension_reducer.fit_transform(features)

        # Predict clusters
        clusters = self.clustering_model.fit_predict(reduced_features)

        return clusters