from typing import Optional

from PyQt5 import QtWebEngineWidgets
from PyQt5 import QtWidgets
from plotly.graph_objs._figure import Figure as PlotlyFigure


class PlotWebWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        QtWidgets.QVBoxLayout(self).addWidget(self.browser)

    def show_graph(self, fig: Optional[PlotlyFigure]):
        if fig:
            html = fig.to_html(include_plotlyjs='cdn')
            self.browser.setHtml(html)
