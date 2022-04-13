import sys
import os
from file_management.load_file import read_excel
from dialogs.generate_dialog import *
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui

from PyQt5.QtCore import pyqtSlot, QEventLoop
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from nrf_setting import ModelSetting

from nrf_coverage_rate import CoverageRate
from nrf_crash_visualization import CrashVisualization
from nrf_attacks import Attacks
from stdout_redirect import StdoutRedirect
# SHOT_SIZE, SHOT_ANGLE, SUB_OBJ = 0, 1, 2
# data_labels = {0: ("", "extreme close-up", "close-up", "middle", "full", "long", "extreme long"), 1: ("", "high", "eye", "low"), 2: ("", "obj", "sub")}

# check detail : 커버리지 측정 표 수치 알아서 나오도록 

form_class=uic.loadUiType("main.ui")[0]
  

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.left=10
        self.top=10
        self.title='딥러닝 모델 취약점 분석 프로그램'
        self.width=1280
        self.height=800
        self.setupUi(self)
        self.setWindowTitle(self.title)
        self.attackData = None
        
        self.attacks = set()
        self.fuzzingCoverages = set()
        self.currentLabel = self.selectedLabelBox.currentText()
        self.dimension = int(self.dimensionBox.currentText())
        self.dataset = ''
        self.selectedAttack = self.attack_combobox_selected()
        self.selectedFuzzingCoverage = self.fuzzing_combobox_selected()
        
        self.colors = {'nc': 'navy', 'kmnc': 'green', 'nbc': 'deeppink', 
                       'fgsm_0.2': 'red', 'pgd_0.2': 'black'} 

        # model & dataset 관련
        
        self.addToolBar(NavigationToolbar(self.GraphWidget.canvas, self))
        self.loadFileBtn.clicked.connect(self.loadFile_clicked)
        self.loadFileBtn_2.clicked.connect(self.loadFile2_clicked)
        
        self.model_combobox.currentIndexChanged.connect(self.model_combobox_selected)
        self.attack_combobox.currentIndexChanged.connect(self.attack_combobox_selected)
        self.fuzzing_combobox.currentIndexChanged.connect(self.fuzzing_combobox_selected)
        self.attackBtn.clicked.connect(self.attackBtn_clicked)
        self.fuzzingBtn.clicked.connect(self.fuzzingBtn_clicked)
        self.graphChangeBtn.clicked.connect(self.graphChangeBtn_clicked)
        
        self.attack_fgsm_checkbox.clicked.connect(self.analysis_checked)
        self.attack_pgd_checkbox.clicked.connect(self.analysis_checked)
        self.fuzzing_nc_checkbox.clicked.connect(self.analysis_checked)
        self.fuzzing_nbc_checkbox.clicked.connect(self.analysis_checked)
        self.fuzzing_kmnc_checkbox.clicked.connect(self.analysis_checked)
    
        self.coverageRateBtn.clicked.connect(self.coverageRate_clicked)
        self.PCAGraphBtn.clicked.connect(self.PCAGraph_clicked)
        # self.defenseBtn.clicked.connecT(self.defense_clicked)
        self.selectedLabelBox.currentIndexChanged.connect(self.label_changed)
        self.dimensionBox.currentIndexChanged.connect(self.dimension_combobox_selected)
        
        self.stdout = StdoutRedirect()
        self.stdout.start()
        self.stdout.printOccur.connect(lambda x: self.append_text(x))
        
        self.dimensionBox.setDisabled(True)
        self.selectedLabelBox.setDisabled(True)
        self.graphChangeBtn.setDisabled(True)
        self.crashVisualization = None
        self.coverageRate = None
        self.method = ''
        
    @pyqtSlot()
    def loadFile_clicked(self):
        fname=QFileDialog.getOpenFileName()
        if fname[0]:
            fileName=os.path.basename(fname[0])
            self.fileNameLabel.setText(fileName)
            filePath=os.path.splitext(fname[0])
            if filePath[-1]=='.xlsx':
                self.data= read_excel(fname[0])
                print(self.data)
                
    @pyqtSlot()
    def loadFile2_clicked(self):
        fname=QFileDialog.getOpenFileName()
        if fname[0]:
            fileName=os.path.basename(fname[0])
            self.datasetNameLabel.setText(fileName)
            filePath=os.path.splitext(fname[0])

    def model_combobox_selected(self):
        if self.model_combobox.currentText() == 'lenet5 & mnist':
            self.dataset = 'mnist'
        elif self.model_combobox.currentText() == 'vgg19 & cifar10':
            self.dataset = 'cifar10'
        elif self.model_combobox.currentText() == 'micronet & GTSRB':
            self.dataset = 'gtsrb' 
        self.attack_combobox_selected()
        self.fuzzing_combobox_selected()
        self.analysis_checked()
        print(self.dataset, " loaded")
        
    def attack_combobox_selected(self):
        self.selectedAttack = self.attack_combobox.currentText().lower()
        if self.selectedAttack == '':
            self.attackBtn.setDisabled(True)
        else:
            self.attackBtn.setEnabled(True)
        if self.model_combobox.currentText() == '':
            self.attackBtn.setDisabled(True)
        
    def fuzzing_combobox_selected(self):
        self.selectedFuzzingCoverage = self.fuzzing_combobox.currentText().lower()
        if self.selectedFuzzingCoverage == '':
            self.fuzzingBtn.setDisabled(True)
        else:
            self.fuzzingBtn.setEnabled(True)
        if self.model_combobox.currentText() == '':
            self.fuzzingBtn.setDisabled(True)
            
    def attackBtn_clicked(self):
        print(self.selectedAttack + " 공격 실행")
        generate_messagebox(self, self.selectedAttack, '공격을 실행합니다.')
        self.attackData = Attacks(self.dataset, self.selectedAttack) 
        self.attackData.execute_attack()
        if self.selectedAttack == "fgsm_0.2":
            self.attack_fgsm_checkbox.setEnabled(True)
        elif self.selectedAttack == "pgd_0.2":
            self.attack_pgd_checkbox.setEnabled(True)
        self.method = 'image'
        
        self.selectedLabelBox.clear()
        self.selectedLabelBox.addItems([str(num) for num in range(len(self.attackData.testImage))])
        self.currentLabel = self.selectedLabelBox.currentText()
        self.selectedLabelBox.setEnabled(True)
        self.graphChangeBtn.setEnabled(True)
        
        self.draw_graph()
                
    def fuzzingBtn_clicked(self):
        print(self.selectedFuzzingCoverage + " 퍼징 실행")
        generate_messagebox(self, self.selectedFuzzingCoverage, '퍼징을 실행합니다.')
        if self.selectedFuzzingCoverage == "nc":
            self.fuzzing_nc_checkbox.setEnabled(True)
        elif self.selectedFuzzingCoverage == "kmnc":
            self.fuzzing_kmnc_checkbox.setEnabled(True)
        elif self.selectedFuzzingCoverage == "nbc":
            self.fuzzing_nbc_checkbox.setEnabled(True)
    
    def analysis_checked(self):
        if self.attack_fgsm_checkbox.isChecked(): self.attacks.add('fgsm_0.2')
        else: self.attacks.discard('fgsm_0.2')
            
        if self.attack_pgd_checkbox.isChecked(): self.attacks.add('pgd_0.2')
        else: self.attacks.discard('pgd_0.2')
            
        if self.fuzzing_nc_checkbox.isChecked(): self.fuzzingCoverages.add('nc')
        else: self.fuzzingCoverages.discard('nc')
        
        if self.fuzzing_nbc_checkbox.isChecked(): self.fuzzingCoverages.add('nbc')
        else: self.fuzzingCoverages.discard('nbc')
        
        if self.fuzzing_kmnc_checkbox.isChecked(): self.fuzzingCoverages.add('kmnc')
        else: self.fuzzingCoverages.discard('kmnc')
        
        if (self.attacks or self.fuzzingCoverages) and self.dataset: 
            self.coverageRateBtn.setEnabled(True)
            self.PCAGraphBtn.setEnabled(True)
        else: 
            self.coverageRateBtn.setEnabled(False)
            self.PCAGraphBtn.setEnabled(False)
     
    def label_changed(self):
        self.currentLabel = self.selectedLabelBox.currentText()

    def coverageRate_clicked(self):
        if not self.coverageRate:
            self.coverageRate = CoverageRate(self.dataset, (self.attacks | self.fuzzingCoverages) )
            generate_messagebox(self, self.selectedFuzzingCoverage, ' 실행합니다.')
            self.coverageRate.measure_neuronMetrics()
            self.coverageRate.measure_attackMetrics()
        self.coverageRate.print_metrics()
            
        self.method = 'coverage'    
        self.draw_graph()
            
        self.selectedLabelBox.clear()
        self.selectedLabelBox.addItems(list(self.coverageRate.bitMetrics))
        
        self.dimensionBox.setDisabled(True)
        self.selectedLabelBox.setDisabled(True)
        self.graphChangeBtn.setEnabled(True)
        
    def PCAGraph_clicked(self):
        if not self.crashVisualization:
            self.crashVisualization = CrashVisualization(self.dataset, (self.attacks | self.fuzzingCoverages))
            generate_messagebox(self, self.selectedFuzzingCoverage, ' 실행합니다.')
            self.crashVisualization.measure_neuronMetrics()
            self.crashVisualization.measure_attackMetrics()
        self.crashVisualization.print_results((self.attacks | self.fuzzingCoverages))
        
        numLabels = self.crashVisualization.modelSetting.numLabels    
        self.selectedLabelBox.clear()
        self.selectedLabelBox.addItems([str(num) for num in range(numLabels)])
            
        self.method = 'pca'
        self.draw_graph()

        self.dimensionBox.setEnabled(True)
        self.selectedLabelBox.setEnabled(True)
        self.graphChangeBtn.setEnabled(True)
            
    def dimension_combobox_selected(self):
        self.dimension = int(self.dimensionBox.currentText())
    
    def graphChangeBtn_clicked(self):
        self.draw_graph()
    
    def draw_graph(self):

        self.GraphWidget.init_graph()
        if self.method == 'pca':
            source = int(self.currentLabel)
            if self.dimension == 2:
                self.GraphWidget.canvas.axes = self.GraphWidget.fig.add_subplot(1,1,1)
                x_points = dict()
                y_points = dict()
                for metric in list(self.attacks | self.fuzzingCoverages):
                    if not len(self.crashVisualization.pcas[source][metric]): continue
                    x_points[metric] = self.crashVisualization.pcas[source][metric][:, 0]
                    y_points[metric] = self.crashVisualization.pcas[source][metric][:, 1]
                    self.GraphWidget.canvas.axes.scatter(x_points[metric], y_points[metric], label = metric, color=self.colors[metric], s=15, alpha=1, marker='+', linewidth=0.5)    
            elif self.dimension == 3:
                self.GraphWidget.canvas.axes = self.GraphWidget.fig.gca(projection='3d')
                x_points = dict()
                y_points = dict()
                z_points = dict()
                for metric in list(self.attacks | self.fuzzingCoverages):
                    x_points[metric] = self.crashVisualization.pcas[source][metric][:, 0]
                    y_points[metric] = self.crashVisualization.pcas[source][metric][:, 1]
                    z_points[metric] = self.crashVisualization.pcas[source][metric][:, 2]   
                    self.GraphWidget.canvas.axes.scatter(x_points[metric], y_points[metric], z_points[metric], label = metric, color = self.colors[metric], s = 1, alpha = 0.5 )
                self.GraphWidget.canvas.axes.view_init(30, 0)
                self.GraphWidget.canvas.axes.set_zlim([-50, 50])
            self.GraphWidget.canvas.axes.set_xlim([-50, 50])
            self.GraphWidget.canvas.axes.set_ylim([-50, 50])
            
        elif self.method == 'coverage':
            self.GraphWidget.canvas.axes = self.GraphWidget.fig.add_subplot(1,1,1)
            for i, metric in enumerate((self.attacks | self.fuzzingCoverages)):
                coverageResults = []
                for metric2 in self.coverageRate.bitMetrics:
                    coverageResult = round(float(self.coverageRate.coverageQueue[metric][metric2].total_cov - np.count_nonzero(self.coverageRate.coverageQueue[metric][metric2].virgin_bits == 0xFF)) * 100 / self.coverageRate.coverageQueue[metric][metric2].total_cov, 2)
                    print('%s found %s coverage of %s' % (metric, metric2, coverageResult))
                    coverageResults.append(coverageResult)
                x = np.arange(len(self.coverageRate.bitMetrics))
                self.GraphWidget.canvas.axes.bar(x + i * 0.1, coverageResults, width = 0.1, label = metric, color=self.colors[metric])
         
            self.GraphWidget.canvas.axes.set_xticks(x) 
            self.GraphWidget.canvas.axes.set_xticklabels(list(self.coverageRate.bitMetrics))
        
        elif self.method == 'image':
            source = int(self.currentLabel)
            if self.attackData:
                self.GraphWidget.canvas.axes1 = self.GraphWidget.fig.add_subplot(1,3,1)
                self.GraphWidget.canvas.axes1.imshow(self.attackData.testImage[source], cmap='gray')
                self.GraphWidget.canvas.axes2 = self.GraphWidget.fig.add_subplot(1,3,2)
                self.GraphWidget.canvas.axes2.imshow(self.attackData.adversarialExamples[source], cmap='gray')
                self.GraphWidget.canvas.axes3 = self.GraphWidget.fig.add_subplot(1,3,3)
                self.GraphWidget.canvas.axes3.imshow(self.attackData.testImage[source] - self.attackData.adversarialExamples[source], cmap='gray')
            
        self.GraphWidget.canvas.axes.spines['right'].set_visible(False)
        self.GraphWidget.canvas.axes.spines['top'].set_visible(False)
          #     self.GraphWidget.canvas.axes.set_zlabel("subj-obj")
        self.GraphWidget.canvas.axes.legend()
        self.GraphWidget.canvas.draw()        
    
    def append_text(self, msg):
        self.textBrowser.moveCursor(QtGui.QTextCursor.End)
        # self.textBrowser.setText(msg)
        self.textBrowser.insertPlainText(msg)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    
if __name__=="__main__":
    app=QApplication(sys.argv)
    myWindow=MyWindow()
    myWindow.show()
    app.exec_()
    
