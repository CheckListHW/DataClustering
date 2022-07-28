import os

import numpy as np
import pandas as pd
import seaborn as sns

import plotly.graph_objs as go
from plotly.graph_objs._figure import Figure as plotlyFigure

from matplotlib import pyplot as plt, cm
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from mycolorpy import colorlist as mcp
from s_dbw import S_Dbw
from sklearn import cluster

from utils.sk_som.example.example import predictions
from utils.sk_som.sklearn_som.som import SOM

if not os.path.exists("../images"): os.mkdir("../images")


def sdbw_choice(X, highest, distances, type_model) -> plotlyFigure:
    '''
    Функция строит метрики оценки кластеров

    df - pd.DataFrame парметров
    highest- верхняя граница количества кластеров
    distances - list список метрик, согласно которой будет подбираться количество кластеров
    type_model - str, 'som' or 'km' or 'agg'
    '''
    range_n_clusters = np.arange(2, highest, 1)

    fig = go.Figure()
    modes = 'markers+lines'

    for d, dtem in enumerate(distances):
        SDbw = []
        for n_clusters in range_n_clusters:

            if type_model == 'som':
                model = SOM(m=n_clusters,
                            n=1,
                            dim=1,
                            random_state=42)
                model.fit(X[1].iloc[:, 2].values.reshape(-1, 1))
                predictions = model.predict(X[1].iloc[:, 2].values.reshape(-1, 1))

            if type_model == 'km':
                model_km = cluster.KMeans(
                    n_clusters=n_clusters,
                    tol=0.01,
                    n_init=500,
                    random_state=42)
                predictions = model_km.fit_predict(X[1].iloc[:, 2].values.reshape(-1, 1))

            if type_model == 'agg':
                model_agl = cluster.AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='l1',
                    linkage='average')
                predictions = model_agl.fit_predict(X[1].iloc[:, 2].values.reshape(-1, 1))

            score = S_Dbw(X[1].iloc[:, :2].values,
                          predictions,
                          centers_id=None,
                          method='Halkidi',
                          alg_noise='comb',
                          centr='mean',
                          nearest_centr=True,
                          metric=dtem)  # 'euclidean')

            SDbw.append(score)
            print(f'Разбивка на {str(n_clusters)} кластера')

        print(f'Расчет с использованием метрики {dtem} завершен!')
        y3 = SDbw

        fig.add_trace(go.Scatter(x=range_n_clusters, y=y3, name=f'SDbw  index for {dtem}', mode=modes, showlegend=True))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        return fig


def prepation_data(path: str):
    area_df = pd.read_table(path, sep=' ', header=None)
    X_NAME, Y_NAME, VAL = 'X', 'Y', 'VALUE'
    area_df.columns = [X_NAME, Y_NAME, VAL]

    # predictions = cl.spectal_cluster(3,area_df.values)
    # area_df.VALUE = predictions[0].tolist()

    area_df[X_NAME] = area_df[X_NAME].astype('int32')
    area_df[Y_NAME] = area_df[Y_NAME].astype('int32')
    area_df[VAL] = area_df[VAL].astype('float32')

    # print('Имена столбцов', area_df.columns.tolist())
    scenarios = area_df.columns[2:].tolist()
    # print('Перечень признаков в датасете', scenarios)

    outcomes = area_df[scenarios].values.reshape(1, -1)[0]
    outcomes_vect = list(outcomes)
    sq_all = len(outcomes_vect)

    grid_df = pd.DataFrame(columns=['X', 'Y'])
    grid_df[X_NAME] = area_df[X_NAME].values
    grid_df[Y_NAME] = area_df[Y_NAME].values
    grid_df['outcomes'] = outcomes_vect

    table = grid_df.pivot(index=Y_NAME, columns=X_NAME, values='outcomes')
    table = table.fillna(-1)

    return table, grid_df


def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]

    # Return colormap object.
    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def build_map(prediction, X, name_of_map, isoline, fig: Figure = None) -> Figure:
    sns.set(rc={'figure.dpi': 150})
    sns.set(rc={'figure.figsize': (20, 10)})

    labels = np.unique(predictions)
    cmap = cmap_discretize(cm.Spectral, len(labels))

    fig, axes = plt.subplots(1, 2) if fig is None else (fig, fig.axes)

    axes[0].contour(X[0], 20, levels=isoline, colors='white')
    sns.heatmap(X[0],
                robust=True,
                cmap=cmap,
                # annot = True,
                xticklabels=False,
                yticklabels=False,
                ax=axes[0]
                )
    axes[0].title.set_text('Clustered map of ' + name_of_map)
    axes[0].invert_yaxis()

    result = X[1][['outcomes']].copy()
    result['Cluster'] = prediction
    sns.boxplot(data=result,
                y='outcomes',
                x='Cluster',
                hue='Cluster',
                palette=mcp.gen_color('Spectral', 4),
                dodge=True)

    return fig
    # plt.show()
