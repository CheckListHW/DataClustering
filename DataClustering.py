import os
import sys

from traceback import format_exception

from MVC.Views.input_view import InputView
from PyQt5.QtWidgets import QApplication

# импорты который не может найти pyinstaller
import sklearn.utils._typedefs
import sklearn.neighbors._partition_nodes
import sklearn.datasets.data
import sklearn.datasets.descr


os.environ['project'] = os.getcwd()


def console_excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(format_exception(exc_type, exc_value, exc_tb))
    print("error!:", tb)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    sys.excepthook = console_excepthook
    window = InputView()
    window.show()
    sys.exit(app.exec_())
