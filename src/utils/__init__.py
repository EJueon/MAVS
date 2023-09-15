from typing import List

import sys
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject

def read_excel(fname: str) -> List:
    df = pd.read_excel(fname, 
                    header=1,
                    usecols="C:F",
                    convert_float=True)
    df = df.dropna()
    arr = df.to_numpy()
    arr = arr.tolist()
    return arr

class StdoutRedirect(QObject):
    printOccur = pyqtSignal(str, str, name = "print")
    
    def __init__(self, *param):
        QObject.__init__(self, None)
        self.sysstdout = sys.stdout.write
        self.sysstderr = sys.stderr.write
    
    def stop(self):
        sys.stdout.write = self.sysstdout
        sys.stderr.write = self.sysstderr
    
    def start(self):
        sys.stdout.write = self.write
        sys.stderr.write = lambda msg: self.write(msg, color = "red")
    
    def write(self, s, color = "black"):
        sys.stdout.flush()
        self.printOccur.emit(s, color)
  

def generate_messagebox(object, title: str, message: str):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(title)
    msg.setInformativeText(message)
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Cancel)
    cancel_button = msg.button(QMessageBox.Cancel)
    cancel_button.setText('확안하였습니다')
    return msg.exec_()
    