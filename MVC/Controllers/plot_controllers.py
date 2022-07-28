from PyQt5.QtWidgets import QGridLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotController(FigureCanvasQTAgg):
    def __init__(self, parent):
        FigureCanvasQTAgg.__init__(self, Figure(tight_layout=True))
        self.mainLayout = QGridLayout(parent)
        self.mainLayout.addWidget(self)
        # self.mainLayout.addWidget(NavigationToolbar2QT(self, parent))

        self.ax = self.figure.add_subplot()
