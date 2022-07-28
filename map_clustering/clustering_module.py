import os
import pickle
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
from matplotlib.colors import ListedColormap
from plotly.subplots import make_subplots
from s_dbw import S_Dbw
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm

if not os.path.exists("../images"): os.mkdir("../images")

np.random.seed(42)
warnings.filterwarnings("ignore")


def build_map(df, facia_names, colors_map, n, clusters_prob):
    f_names = facia_names.copy()
    f_names.insert(0, 'None')
    cmap = ListedColormap(colors_map)
    fig, ax = plt.subplots()

    a = ax.pcolormesh(df.loc[n]['script_states_map'], cmap=cmap)

    fig.set_figwidth(12)
    fig.set_figheight(12)
    cbar = plt.colorbar(a, ticks=range(-1, 10))
    cbar.ax.set_yticklabels(f_names)
    plt.grid()
    plt.title('Карта фаций сценария № ' + str(n) + '.\n  Правдоподобие сценария = ' \
              + str(round(df.loc[n]['script_prob'], 3)) \
              + '\n Вероятность кластера = ' + str(clusters_prob))

    plt.show()


def build_all_maps(df, facia_names, colors_map, labels, centroid_indexes):
    df['labels'] = labels[0]
    # centroid_indexes = spec_centroid_indexes
    ind = [df.iloc[i].labels for i in centroid_indexes]
    clusters_size = dict(Counter(labels[0]))
    clusters_probs = [clusters_size[i] / len(labels[0]) for i in ind]
    max_ver_indexes = [df[df.labels == i].script_prob.idxmax() for i in ind]
    print('КАРТЫ С МАКСИМАЛЬНЫМ ПРАВДОПОДОБИЕМ')
    for i, item in enumerate(ind):
        build_map(df, facia_names, colors_map, max_ver_indexes[i], clusters_probs[i])
    print('ЦЕНТРОИДНЫЕ КАРТЫ')
    for i, item in enumerate(ind):
        build_map(df, facia_names, colors_map, centroid_indexes[i], clusters_probs[i])


def choose_colors(scenarios):
    colors = ['#AA0DFE',
              '#3283FE',
              '#85660D',
              '#782AB6',
              '#565656',
              '#1C8356',
              '#16FF32',
              '#F7E1A0',
              '#E2E2E2',
              '#1CBE4F',
              '#C4451C',
              '#DEA0FD',
              '#FE00FA',
              '#325A9B',
              '#FEAF16',
              '#F8A19F',
              '#90AD1C',
              '#FBE426',
              '#1CFFCE',
              '#2ED9FF',
              '#B10DA1',
              '#C075A6']
    l = len(scenarios)
    if l > len(colors):
        raise AttributeError("Фаций заявлено больше, чем предусматривалось")

    return colors[:l]


def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def GC(df, col_of_label='label'):
    '''
    Функция возвращает индексы центроидов в датасете df
    df - датасет с фичиами
    col_of_label - имя столбца, в котором указаны метки кластера
    '''
    closest = []
    n_clusters = len(np.unique(df[col_of_label]))
    for cl in range(n_clusters):
        df_where_only_i_cluster = df[df.label == cl].drop(columns=df.columns[-1])
        centroid = df_where_only_i_cluster.mean()
        centroid = np.array(centroid).reshape(1, -1)
        centroid, _ = pairwise_distances_argmin_min(centroid, df_where_only_i_cluster)
        closest.append(df_where_only_i_cluster.iloc[centroid].index[0])
    return closest


# def get_centroid_index(labels):
#     '''
#     id_spaces - список имен пространств кластеризации, которые использовались для кластеризации:
#         1) segment_probability,2) segment_probability_dissection,
#         3) segment_probability_dissection_tr_matrix, 4) featurelist_new1,
#         5) featurelist_new2, 6) featurelist_new3
#
#     spaces - список массив пространств признаков, которые используются для кластеризации:
#         1) segment_probability  - исходное без вектора расчлененности
#         2) segment_probability_dissection - исходное + вектор расчлененности
#         3) segment_probability_dissection_tr_matrix - исходное + вектор расчлененности + матрица площадей соседства
#         4) featurelist_new1 исходное сжатое без вектора расчлененности
#         5) featurelist_new2  исходное сжатое + вектор расчлененности сжатие вместе с вектором расчлененности
#         6) featurelist_new3  исходное сжатое вектор расчлененности + матрица площадей соседства сжатие вместе с вектором расчлененности и матрицей площадей
#
#     labels - список списков, внутри которых метки кластеров для каждого исхода
#
#     return
#         Функция возвращает словарь, где ключи - название пространства кластеризации, а значения - индексы центроидов
#     '''
#     indexes = dict()
#     for j, jtem in enumerate(labels):
#         clustering_result = pd.DataFrame(spaces[j], dtype='float32')
#
#         clustering_result['label'] = jtem
#         clustering_result['label'] = clustering_result['label'].astype('float32')
#         print(j)
#         print('Для пространства признаков ', id_spaces[j])
#         print('Индексы центроидных сценариев ', GC(clustering_result, 'label'))
#         print()
#         item = {id_spaces[j]: GC(clustering_result, 'label')}
#         indexes = merge_two_dicts(indexes, item)
#
#         del [[clustering_result]]
#         gc.collect()
#         clustering_result = pd.DataFrame()
#     return indexes


def one_hot_encoder(x, value, sq_seg, num):
    one_hot_vector = np.zeros(num)
    one_hot_vector[x] = value * sq_seg  # round(value*sq_seg,4)
    return one_hot_vector


def space_creation(df, space_name):
    def f1():
        return space.append(np.array(features_list).flatten())

    def f2():
        return space.append(np.concatenate([np.array(features_list).flatten(),  # Характеристики сегментов в сценарии
                                            row['facies_dissection'],  # Вектор расчленнности
                                            row['sq_no_corr_nbh']  # Площадь некорректного соседства в сценарии
                                            ]
                                           )
                            )

    def f3():
        return space.append(np.concatenate([np.array(features_list).flatten(),
                                            row['facies_dissection'],
                                            row['sq_no_corr_nbh'],
                                            row['script_tr_matrix'] / np.sum(row['script_tr_matrix']).flatten() / num
                                            ]
                                           )
                            )

    def func(i):
        return {'segment_probabilities': f1,
                'segment_probability_dissection': f2,
                'segment_probability_dissection_tr_matrix': f3}.get(i, 'error')

    if space_name == 'featurelist_new1':
        space_name = 'segment_probabilities'
    if space_name == 'featurelist_new2':
        space_name = 'segment_probability_dissection'
    if space_name == 'featurelist_new3':
        space_name = 'segment_probability_dissection_tr_matrix'

    '''
    df - датасет для кластеризации
    
    '''
    num = len(df['script_tr_matrix'][0])
    space = []

    for i, row in df.iterrows():
        features_list = row['script_feature_segment'].copy()
        for j in range(len(features_list)):
            features_list[j] = one_hot_encoder(features_list[j][0], features_list[j][1], features_list[j][2], num)
        func(space_name)()

    return space


def n_dim_for_data(df1):
    N_COMP = []
    n_comp = [4, 5, 10, 15, 20, 25, 30, 35, 40]  # list containing different values of components
    explained = []  # explained variance ratio for each component of Truncated SVD

    for x in n_comp:
        svd = TruncatedSVD(n_components=x)
        svd.fit(df1)
        if 0.93 >= svd.explained_variance_ratio_.sum() <= 0.97:
            N_COMP.append(x)

    '''
    #print(explained)  
    N_COMP = np.where(np.array(explained) > 0.93)[0].tolist()
    
    for e,etem in enumerate(np.array(explained)):
        print(etem)
        if (etem >= 0.93) == True and (etem <= 0.97) == True:
            print(e)
            N_COMP.append(n_comp[e])
    #print(explained)  
    '''
    return int(np.mean(N_COMP))


def low_dim_creator(n_comp, df1):
    '''
    n_comp - осредненное количество компонент нового низкоразмерного пространства
    df1,d2,df3 список массив пространств признаков, которые используются для кластеризации:
        1) df1 is segment_probability  - исходное без вектора расчлененности
        2) df2 is segment_probability_dissection - исходное + вектор расчлененности 
        3) df3 is segment_probability_dissection_tr_matrix - исходное + вектор расчлененности + матрица площадей соседства
    '''
    svd1 = TruncatedSVD(n_components=n_comp, n_iter=7, random_state=42)
    featurelist_new1 = svd1.fit_transform(df1)
    return featurelist_new1


def spectal_cluster(number_clusters, space):
    labels = []
    spectral = SpectralClustering(
        n_clusters=number_clusters,
        affinity='nearest_neighbors',
        assign_labels='discretize',
        random_state=42).fit(space)
    labels.append(spectral.labels_)
    return labels


def agglomerative_cluster(number_clusters, space, metrics):
    labels_agg = []
    agglomerative = AgglomerativeClustering(
        n_clusters=number_clusters,
        affinity=metrics,
        linkage='average').fit(pd.DataFrame(np.array(space)))
    labels_agg.append(agglomerative.labels_)
    return labels_agg


def color_generator(number_clusters):
    sns.color_palette('deep', n_colors=number_clusters)
    color = list(sns.color_palette(n_colors=number_clusters).as_hex())
    return color


def TNSE_embedded_spaces(spaces, perp, metrics):
    feature_embedded_spaces = []
    for j, jtem in tqdm(enumerate(spaces)):
        feature_embedded = TSNE(n_components=3,
                                perplexity=perp,
                                metric=metrics,
                                square_distances=True,
                                random_state=42,
                                method='exact').fit_transform(jtem)

        feature_embedded_spaces.append(feature_embedded)
    return feature_embedded_spaces


def cluster_visualizer(dfs, colors, labels, centroids, df, clustering_type='"no data"', sample_index=None):
    size = 8
    # dim1,dim2 = 1,2
    clusters_size = dict(Counter(labels[0]))
    clusters_probs = [clusters_size[size] / len(labels[0]) for size in clusters_size.keys()]
    clusters_probs = dict(zip(clusters_size.keys(), clusters_probs))
    df['labels'] = labels[0]
    max_ver_indexes = [df[df.labels == i].script_prob.idxmax() for i in clusters_size.keys()]

    for j, jtem in enumerate(labels):
        fig = go.Figure(
            data=go.Scatter3d(
                x=dfs[j][:, 0],
                y=dfs[j][:, 1],
                z=dfs[j][:, 2],
                mode='markers',
                showlegend=False,
                marker=dict(
                    size=size,
                    color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0]),
                    opacity=0.8)
            ))
        fig.add_scatter3d(x=dfs[j][:, 0][max_ver_indexes],
                          y=dfs[j][:, 1][max_ver_indexes],
                          z=dfs[j][:, 2][max_ver_indexes],
                          showlegend=False,
                          mode='markers', marker=dict(symbol='diamond',
                                                      color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                          max_ver_indexes],
                                                      size=8,
                                                      opacity=1,
                                                      ))

        fig.add_scatter3d(x=dfs[j][:, 0][centroids],
                          y=dfs[j][:, 1][centroids],
                          z=dfs[j][:, 2][centroids],
                          showlegend=False,
                          mode='markers', marker=dict(symbol='x',
                                                      color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                          centroids],
                                                      size=6,
                                                      opacity=1))
        if sample_index:
            fig.add_scatter3d(x=[dfs[j][:, 0][sample_index]],
                              y=[dfs[j][:, 1][sample_index]],
                              z=[dfs[j][:, 2][sample_index]],
                              showlegend=False,
                              mode='markers', marker=dict(color='red',
                                                          size=10,
                                                          opacity=1))

        fig.show()

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("XY", "XZ", "YZ")
                            )

        f1 = go.Scatter(
            x=dfs[j][:, 0],
            y=dfs[j][:, 1],
            mode='markers',
            hovertext=jtem,
            showlegend=False,
            marker=dict(
                size=size,
                color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0]),
                opacity=0.8))

        f11 = go.Scatter(x=dfs[j][:, 0][max_ver_indexes],
                         y=dfs[j][:, 1][max_ver_indexes],
                         showlegend=False,
                         mode='markers', marker=dict(symbol='diamond',
                                                     color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                         max_ver_indexes],
                                                     size=10,
                                                     opacity=1,
                                                     ))

        f111 = go.Scatter(x=dfs[j][:, 0][centroids],
                          y=dfs[j][:, 1][centroids],
                          showlegend=False,
                          mode='markers', marker=dict(symbol='x',
                                                      color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                          centroids],
                                                      size=12,
                                                      opacity=1))

        f2 = go.Scatter(
            x=dfs[j][:, 0],
            y=dfs[j][:, 2],
            mode='markers',
            hovertext=jtem,
            showlegend=False,
            marker=dict(
                size=size,
                color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0]),
                opacity=0.8)
        )

        f22 = go.Scatter(x=dfs[j][:, 0][max_ver_indexes],
                         y=dfs[j][:, 2][max_ver_indexes],
                         showlegend=False,
                         mode='markers', marker=dict(symbol='diamond',
                                                     color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                         max_ver_indexes],
                                                     size=10,
                                                     opacity=1,
                                                     ))

        f222 = go.Scatter(x=dfs[j][:, 0][centroids],
                          y=dfs[j][:, 2][centroids],
                          showlegend=False,
                          mode='markers', marker=dict(symbol='x',
                                                      color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                          centroids],
                                                      size=12,
                                                      opacity=1))

        f3 = go.Scatter(
            x=dfs[j][:, 1],
            y=dfs[j][:, 2],
            mode='markers',
            hovertext=jtem,
            showlegend=False,
            marker=dict(
                size=size,
                color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0]),
                opacity=0.8)
        )

        f33 = go.Scatter(x=dfs[j][:, 1][max_ver_indexes],
                         y=dfs[j][:, 2][max_ver_indexes],
                         showlegend=False,
                         mode='markers', marker=dict(symbol='diamond',
                                                     color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                         max_ver_indexes],
                                                     size=10,
                                                     opacity=1,
                                                     ))

        f333 = go.Scatter(x=dfs[j][:, 1][centroids],
                          y=dfs[j][:, 2][centroids],
                          showlegend=False,
                          mode='markers', marker=dict(symbol='x',
                                                      color=pd.Series(jtem).map(pd.DataFrame(colors).to_dict()[0])[
                                                          centroids],
                                                      size=12,
                                                      opacity=1))

        fig.add_trace(
            f1,
            row=1, col=1
        )
        fig.add_trace(
            f11,
            row=1, col=1
        )
        fig.add_trace(
            f111,
            row=1, col=1
        )
        fig.add_trace(
            f2,
            row=1, col=2
        )
        fig.add_trace(
            f22,
            row=1, col=2
        )
        fig.add_trace(
            f222,
            row=1, col=2
        )
        fig.add_trace(
            f3,
            row=1, col=3
        )
        fig.add_trace(
            f33,
            row=1, col=3
        )
        fig.add_trace(
            f333,
            row=1, col=3
        )

        if sample_index:
            F1 = go.Scatter(x=[dfs[j][:, 0][sample_index]],
                            y=[dfs[j][:, 1][sample_index]],
                            showlegend=False,
                            mode='markers', marker=dict(color='red',
                                                        size=10,
                                                        opacity=1))

            F2 = go.Scatter(x=[dfs[j][:, 0][sample_index]],
                            y=[dfs[j][:, 2][sample_index]],
                            showlegend=False,
                            mode='markers', marker=dict(color='red',
                                                        size=10,
                                                        opacity=1))
            F3 = go.Scatter(x=[dfs[j][:, 1][sample_index]],
                            y=[dfs[j][:, 2][sample_index]],
                            showlegend=False,
                            mode='markers', marker=dict(color='red',
                                                        size=10,
                                                        opacity=1))
            fig.add_trace(
                F1,
                row=1, col=1
            )
            fig.add_trace(
                F2,
                row=1, col=2
            )
            fig.add_trace(
                f3,
                row=1, col=3
            )

        fig.update_layout(
            title={'text': f"Метод кластеризации:{clustering_type} / Вероятности кластеров: {str(clusters_probs)}",
                   'y': 0.9,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            width=1000,
            height=600)

        fig.show()


def centroid_indexes(dfs, labels):
    indexes = dict()
    for j, jtem in enumerate(labels):
        clustering_result = pd.DataFrame(dfs[j], dtype='float32')
        clustering_result['label'] = jtem
        clustering_result['label'] = clustering_result['label'].astype('float32')

        item = GC(clustering_result, 'label')
        # indexes = merge_two_dicts(indexes,item)

    return item


def sdbw_choice(df, highest, distances):
    from sklearn.metrics import pairwise_distances
    '''
    Функция строит метрики оценки кластеров
    
    df - pd.DataFrame парметров
    highest- верхняя граница количества кластеров
    distances - list список метрик, согласно которой будет подбираться количество кластеров    
    '''
    range_n_clusters = np.arange(2, highest, 1)

    fig = go.Figure()
    modes = 'markers+lines'

    for d, dtem in enumerate(distances):
        SDbw = []
        print(dtem)
        for n_clusters in range_n_clusters:
            # clusterer = AgglomerativeClustering(
            #    n_clusters = n_clusters,
            #    affinity = dtem,
            #    linkage='average').fit(df)

            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='discretize').fit(pairwise_distances(df, metric=dtem))

            score = S_Dbw(np.array(df),
                          clusterer.labels_,
                          centers_id=None,
                          method='Halkidi',
                          alg_noise='comb',
                          centr='mean',
                          nearest_centr=True,
                          metric=dtem)  # 'euclidean')

            SDbw.append(score)

        y3 = SDbw

        fig.add_trace(
            go.Scatter(
                x=range_n_clusters,
                y=y3,
                # showlegend = False,
                name='SDbw  index for  ' + dtem,
                mode=modes,
                showlegend=True,
                # marker = dict(color = 'blue'),
            )
        )

        fig.update_layout(
            # title = ' DISTANCE IS ' + dtem,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    fig.show()


def clustering_module(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)

    # if not flag:
    df = data['df_samples']
    # else:
    # df = data['df_samples'].iloc[flag].reset_index(drop = False)

    facia_names = data['scenarios'].copy()

    colors_map = ['black'] + choose_colors(facia_names)

    sdbw_metric = ["braycurtis",
                   "canberra",
                   "chebyshev",
                   "cityblock",
                   "correlation",
                   "cosine",
                   "dice",
                   "euclidean",
                   "hamming",
                   "jaccard"]

    id_spaces = [  # 'segment_probabilities',
        'segment_probability_dissection',
        # 'segment_probability_dissection_tr_matrix',
        # 'featurelist_new1',
        'featurelist_new2',
        # 'featurelist_new3'
    ]
    use_pre_set = None
    while use_pre_set != 'y' and use_pre_set != 'n' and use_pre_set != 'Y' and use_pre_set != 'N':
        use_pre_set = str(input("ИСПОЛЬЗОВАТЬ PRESET? (Y/N)       "))

    if use_pre_set == 'y' or use_pre_set == 'Y':
        metrics = 'chebyshev'
        space_name = 'featurelist_new2'
        space = space_creation(df, space_name)
        clust_num = 3
        use_low_dim = 'N'
        perp = 25
    else:
        space_name = None
        while space_name not in id_spaces:
            space_name = id_spaces[int(input(
                """Использовать для кластеризации: исходное пространство[0],сжатое пространство[1] (введите индекс)"""))]

        space = space_creation(df, space_name)

        sdbw_choice(space, 15, sdbw_metric)
        metrics = None
        while metrics not in sdbw_metric:
            metrics = sdbw_metric[int(input("Какую метрику кластеризации использовать?(введите индекс) \n" + ''.join(
                map(str, [i + str([ind]) + '\n' for ind, i in enumerate(sdbw_metric)])) + "     "))]
        clust_num = int(input("ВВЕДИТЕ КОЛИЧЕСТВО КЛАСТЕРОВ   "))
        use_low_dim = None
        while use_low_dim != 'y' and use_low_dim != 'n' and use_low_dim != 'Y' and use_low_dim != 'N':
            use_low_dim = str(input("Понизить размерность для визуализации? (Y/N)       "))

    if use_low_dim == 'y' or use_pre_set == 'Y':
        perp_list = [10, 20, 30, 40, 50]
        for j, perp in enumerate(perp_list):
            feature_embedded = TSNE(n_components=3,
                                    perplexity=perp,
                                    metric=metrics,
                                    square_distances=True,
                                    random_state=42,
                                    method='exact').fit_transform(space)
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=feature_embedded[:, 0],
                    y=feature_embedded[:, 1],
                    z=feature_embedded[:, 2],
                    mode='markers'
                )])
            fig.update_layout(title={'text': f"perplexity = {perp}",
                                     'y': 0.9,
                                     'x': 0.55,
                                     'xanchor': 'center',
                                     'yanchor': 'top'}
                              )
            fig.show()

        perp = float(input("Введите значение perplexity:    "))

    else:
        perp = 25

    spec_labels = spectal_cluster(clust_num, space)
    agglomerative_labels = agglomerative_cluster(clust_num, space, metrics)
    color = color_generator(clust_num)
    embedded_spaces = TNSE_embedded_spaces([space], perp, metrics)
    spec_centroid_indexes = centroid_indexes([space], spec_labels)
    agglomerative_centroid_indexes = centroid_indexes([space], agglomerative_labels)
    cluster_visualizer(embedded_spaces, color, spec_labels, spec_centroid_indexes, df, 'Спектральная')
    cluster_visualizer(embedded_spaces, color, agglomerative_labels, agglomerative_centroid_indexes, df,
                       'Агломеративная')
    print('Спектральная кластеризация')
    build_all_maps(df, facia_names, colors_map, spec_labels, spec_centroid_indexes)
    print('Агломеративная кластеризация')
    build_all_maps(df, facia_names, colors_map, agglomerative_labels, agglomerative_centroid_indexes)

    return spec_labels[0], agglomerative_labels[0], embedded_spaces[0]
