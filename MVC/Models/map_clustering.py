import enum
from typing import Optional

from plotly.graph_objs._figure import Figure as PlotlyFigure
from matplotlib.pyplot import Figure
from sklearn.cluster import AgglomerativeClustering, KMeans

from utils.map_tools import prepation_data, sdbw_choice, build_map
from utils.sk_som.sklearn_som.som import SOM


class SdbwMetric(enum.Enum):
    chebyshev = "chebyshev"
    cityblock = "cityblock"
    cosine = "cosine"
    euclidean = 'euclidean'


class TypeModel(enum.Enum):
    km = "km"
    som = "som"
    agg = "agg"


def get_predictions(type_model: TypeModel, n_clust: int, X):
    if type_model is TypeModel.som:
        model = SOM(m=n_clust, n=1, dim=1, random_state=42)
        model.fit(X[1].iloc[:, 2].values.reshape(-1, 1))
    elif type_model is TypeModel.km:
        model = KMeans(n_clusters=n_clust, tol=0.01, n_init=500, random_state=42)
    elif type_model is TypeModel.agg:
        model = AgglomerativeClustering(n_clusters=n_clust, affinity='cityblock', linkage='average')
    else:
        return None
    return model.fit_predict(X[1].iloc[:, 2].values.reshape(-1, 1))


class MapClustering:
    def __init__(self):
        self.sdbw_figure: Optional[PlotlyFigure] = None
        self.map_figure: Optional[Figure] = None
        self.sdbw_metric: SdbwMetric = SdbwMetric.chebyshev
        self.map = 'Env'
        self.maps_path = {}
        self.type_model = TypeModel.km
        self.n_clust = 4
        self.path = ''

    def start(self):
        self.validation_parametrs()

        X = prepation_data(self.maps_path[self.map])
        self.sdbw_figure = sdbw_choice(X, 10, [self.sdbw_metric.value], self.type_model.value)
        sdbw_choice(X, 10, self.sdbw_metric.value, self.type_model.value)
        predictions = get_predictions(self.type_model, self.n_clust, X)
        self.map_figure = build_map(predictions, X, self.map, [1000], self.map_figure)

    def validation_parametrs(self):
        self.chechk_param(self.sdbw_metric, SdbwMetric)
        self.chechk_param(self.type_model, TypeModel)
        self.chechk_param(self.map, str)
        self.chechk_param(self.n_clust, int)

    def chechk_param(self, param, target):
        if type(param) is not target:
            raise AttributeError(param) from None

    def get_sdbw_figure(self) -> Optional[PlotlyFigure]:
        return self.sdbw_figure

    def get_map_figure(self) -> Optional[Figure]:
        return self.map_figure

