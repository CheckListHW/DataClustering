import re
from os import environ

from PyQt5 import uic
from utils.a_thread import AThread
from PyQt5.QtWidgets import QMainWindow

from MVC.Views.plot_web_widget import PlotWebWidget
from MVC.Controllers.plot_controllers import PlotController
from MVC.Models.map_clustering import MapClustering, SdbwMetric, TypeModel
from utils.file import FileEdit


class InputView(QMainWindow):
    def __init__(self):
        super(InputView, self).__init__()
        uic.loadUi(environ['project'] + '/ui/map_clustering.ui', self)
        self.map_cl = MapClustering()
        self.controller, self.map_controller = PlotController(self.plotFrame), PlotController(self.plotFrame_2)

        self.map_cl.map_figure = self.map_controller.figure

        self.web_widget = PlotWebWidget(self.plotWebFrame)
        self.plotWebVerticalLayout.addWidget(self.web_widget)
        self.handlers_connect()
        # self.debug()

    def debug(self):
        self.map_cl.maps_path['Env'] = 'C:/Users/KosachevIV/PycharmProjects/dataClustering/input/Env'
        self.start_calc()

    def n_clust_change(self):
        value = self.n_clustComboBox.currentText()
        self.map_cl.n_clust = int(value)

    def sdbwMetric_change(self):
        value = self.sdbwMetricComboBox.currentText()
        if hasattr(SdbwMetric, value):
            self.map_cl.sdbw_metric = getattr(SdbwMetric, value)

    def maps_change(self):
        value = self.mapsComboBox.currentText()
        self.map_cl.map = value

    def type_models_change(self):
        value = self.typeModelsComboBox.currentText()
        if hasattr(TypeModel, value):
            self.map_cl.type_model = getattr(TypeModel, value)

    def handlers_connect(self) -> None:
        self.n_clustComboBox.activated.connect(self.n_clust_change)
        self.sdbwMetricComboBox.activated.connect(self.sdbwMetric_change)
        self.mapsComboBox.activated.connect(self.maps_change)
        self.typeModelsComboBox.activated.connect(self.type_models_change)
        self.startButton.clicked.connect(self.start_calc)
        self.addMapFilesButton.clicked.connect(self.add_map_files)

    def add_map_files(self):
        file = FileEdit(self)
        paths = file.open_files('')
        if paths is []:
            return

        for path in paths:
            x = re.split(r'/', path)
            self.map_cl.maps_path[x[-1]] = path

        text = ''
        self.mapsComboBox.clear()
        for k, v in self.map_cl.maps_path.items():
            self.mapsComboBox.addItem(k)
            text += f'{k}:\n {v}\n\n'

        self.pathBrowser.setText(text)
        self.startButton.setEnabled(True)

    def set_info(self):
        pass

    def set_sdbw(self):
        pass

    def set_map(self):
        pass

    def update_plots(self):
        self.web_widget.show_graph(self.map_cl.get_sdbw_figure())

    def start_calc(self):
        self.controlFrame.setEnabled(False)
        self.calc_th = AThread()
        self.calc_th.finished.connect(self.update_plots)
        self.calc_th.callback = self._start_calc
        self.calc_th.start()

    def _start_calc(self):
        self.map_cl.start()
        self.controlFrame.setEnabled(True)

