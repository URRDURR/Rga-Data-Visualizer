from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QTabWidget,QWidget
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("App")

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)
        tabs.setMovable(True)
        tabs.addTab(QWidget(),"test")
        tabs.addTab(QWidget(),"2")
        tabs.addTab(QWidget(),"3")
        tabs.addTab(QWidget(),"4")
        tabs.addTab(QWidget(),"5")

        # Set the central widget of the Window.
        self.setCentralWidget(tabs)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
