from PyQt5.QtWidgets import (
    QLabel, QWidget, QPushButton, QFileDialog, QMessageBox,
    QSlider, QSpinBox, QInputDialog
)
from PyQt5.QtGui import QBrush, QPen, QPainter, QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import os
import cv2 as cv
import numpy as np

import _automatic_
import _semiautomatic_
import resources

global flag


# ──────────────────────────────────────────────────────────────────────────────
# Scroll-to-Edit label: wheel scrolls through slices, double-click opens editor
# ──────────────────────────────────────────────────────────────────────────────
class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._nav = None

    def setNavigator(self, nav):
        self._nav = nav

    def wheelEvent(self, event):
        if self._nav and self._nav.slicesManager.isEnabled():
            if event.angleDelta().y() > 0:
                self._nav.previousSlice()
            else:
                self._nav.nextSlice()
            event.accept()
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._nav and self._nav.slicesManager.isEnabled():
            self._nav.openWindow()
        else:
            super().mouseDoubleClickEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Main window with hotkey support (←→ navigate, E = edit)
# ──────────────────────────────────────────────────────────────────────────────
class MainWindowClass(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._ui = None

    def setUi(self, ui):
        self._ui = ui

    def keyPressEvent(self, event):
        ui = self._ui
        if ui and ui.slicesManager.isEnabled():
            key = event.key()
            if key in (Qt.Key_Right, Qt.Key_Down):
                ui.nextSlice()
                return
            elif key in (Qt.Key_Left, Qt.Key_Up):
                ui.previousSlice()
                return
            elif key == Qt.Key_E:
                ui.openWindow()
                return
        super().keyPressEvent(event)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        'STYLESHEET'
        buttonStyle = """
        QPushButton{
                border-style: solid;
                border-width: 0px;
                border-radius: 10px;
                background-color: rgb(234,234,234);
        }
        QPushButton::hover{
                background-color: rgb(183,181,181);
        }
        """
        groupBoxStyle = """
        QGroupBox{
                background-color:none;
                border: 1px solid rgb(183,181,181);
                border-radius: 10px;
        }
        """

        closeStyle = """
        QPushButton{
            border-style: solid;
            border-width: 0px;
            border-radius: 10px;
            background-color: rgb(234,67,53);
            color:rgb(255, 255, 255)
        }
        QPushButton::hover{
            background-color: rgb(172,16,12);
        }
        """

        generateVolumeStyle = """
        QPushButton{
                border-style: solid;
                border-width: 0px;
                border-radius: 10px;
                background-color: rgb(70,136,244);
                color:rgb(255, 255, 255)
        }

        QPushButton::hover
        {
                background-color: rgb(18,77,150);
        }
        """

        boxStyle = """
        QGroupBox{
            border-style: none;
            background-color:none;
        }
        """

        textEditStyle = """
        QTextEdit{
            border-style: solid;
            border-width: 1px;
            border-radius: 10px;
            border-color: rgb(183,181,181);
        }
        """

        lineEditStyle = """
        QLineEdit{
            border-style: solid;
            border-width: 1px;
            border-radius: 10px;
            border-color: rgb(183,181,181);
        }
        """

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1208, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1208, 700))
        MainWindow.setMaximumSize(QtCore.QSize(1208, 700))
        MainWindow.setMouseTracking(False)
        MainWindow.setTabletTracking(False)
        MainWindow.setStyleSheet("background-color:rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.filename = QtWidgets.QLineEdit(self.centralwidget)
        self.filename.setGeometry(QtCore.QRect(740, 50, 411, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.filename.setFont(font)
        self.filename.setStyleSheet(lineEditStyle)
        self.filename.setFrame(False)
        self.filename.setObjectName("filename")
        self.openFile = QtWidgets.QPushButton(self.centralwidget)
        self.openFile.setGeometry(QtCore.QRect(600, 50, 131, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.openFile.setFont(font)
        self.openFile.setStyleSheet(buttonStyle)
        self.openFile.setObjectName("openFile")

        # ── Slice image (ClickableLabel for scroll + double-click) ────────────
        self.imageHeart = ClickableLabel(self.centralwidget)
        self.imageHeart.setEnabled(True)
        self.imageHeart.setGeometry(QtCore.QRect(30, 20, 520, 520))
        self.imageHeart.setText("")
        self.imageHeart.setPixmap(QtGui.QPixmap(":/aux/default_txt.png"))
        self.imageHeart.setScaledContents(True)
        self.imageHeart.setAlignment(QtCore.Qt.AlignCenter)
        self.imageHeart.setObjectName("imageHeart")

        self.manual_intervention = QtWidgets.QGroupBox(self.centralwidget)
        self.manual_intervention.setEnabled(True)
        self.manual_intervention.setGeometry(QtCore.QRect(590, 320, 591, 211))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        self.manual_intervention.setFont(font)
        self.manual_intervention.setStyleSheet(groupBoxStyle)
        self.manual_intervention.setTitle("")
        self.manual_intervention.setObjectName("manual_intervention")
        self.duringEditionBox = QtWidgets.QGroupBox(self.manual_intervention)
        self.duringEditionBox.setGeometry(QtCore.QRect(0, 15, 571, 91))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        self.duringEditionBox.setFont(font)
        self.duringEditionBox.setStyleSheet(boxStyle)
        self.duringEditionBox.setTitle("")
        self.duringEditionBox.setObjectName("duringEditionBox")
        self.generateVolumeSemi = QtWidgets.QPushButton(self.duringEditionBox)
        self.generateVolumeSemi.setGeometry(QtCore.QRect(430, 0, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.generateVolumeSemi.setFont(font)
        self.generateVolumeSemi.setMouseTracking(False)
        self.generateVolumeSemi.setStyleSheet(generateVolumeStyle)
        self.generateVolumeSemi.setObjectName("generateVolumeSemi")
        self.slices = QtWidgets.QTextEdit(self.duringEditionBox)
        self.slices.setGeometry(QtCore.QRect(130, 3, 281, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.slices.setFont(font)
        self.slices.setStyleSheet(textEditStyle)
        self.slices.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.slices.setReadOnly(True)
        self.slices.setObjectName("slices")
        self.slices_edited = QtWidgets.QLabel(self.duringEditionBox)
        self.slices_edited.setGeometry(QtCore.QRect(30, 10, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.slices_edited.setFont(font)
        self.slices_edited.setObjectName("slices_edited")
        self.infoAuto_2 = QtWidgets.QGroupBox(self.duringEditionBox)
        self.infoAuto_2.setGeometry(QtCore.QRect(320, 50, 251, 41))
        self.infoAuto_2.setStyleSheet(boxStyle)
        self.infoAuto_2.setTitle("")
        self.infoAuto_2.setObjectName("infoAuto_2")
        self.textAuto_2 = QtWidgets.QLabel(self.infoAuto_2)
        self.textAuto_2.setGeometry(QtCore.QRect(30, 10, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(7)
        font.setBold(False)
        font.setWeight(50)
        self.textAuto_2.setFont(font)
        self.textAuto_2.setStyleSheet("background-color:none;")
        self.textAuto_2.setObjectName("textAuto_2")
        self.iconAuto_2 = QtWidgets.QLabel(self.infoAuto_2)
        self.iconAuto_2.setGeometry(QtCore.QRect(10, 10, 15, 15))
        self.iconAuto_2.setAutoFillBackground(False)
        self.iconAuto_2.setText("")
        self.iconAuto_2.setPixmap(QtGui.QPixmap(":/aux/info.png"))
        self.iconAuto_2.setScaledContents(True)
        self.iconAuto_2.setObjectName("iconAuto_2")
        self.afterEditionBox = QtWidgets.QGroupBox(self.manual_intervention)
        self.afterEditionBox.setGeometry(QtCore.QRect(0, 100, 571, 101))
        self.afterEditionBox.setStyleSheet(boxStyle)
        self.afterEditionBox.setTitle("")
        self.afterEditionBox.setObjectName("afterEditionBox")
        self.volume_2 = QtWidgets.QTextEdit(self.afterEditionBox)
        self.volume_2.setGeometry(QtCore.QRect(30, 45, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.volume_2.setFont(font)
        self.volume_2.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.volume_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.volume_2.setReadOnly(True)
        self.volume_2.setObjectName("volume_2")
        self.new_volume = QtWidgets.QLabel(self.afterEditionBox)
        self.new_volume.setGeometry(QtCore.QRect(30, 20, 180, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.new_volume.setFont(font)
        self.new_volume.setObjectName("new_volume")
        self.view3DSemi = QtWidgets.QPushButton(self.afterEditionBox)
        self.view3DSemi.setGeometry(QtCore.QRect(430, 60, 131, 28))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.view3DSemi.setFont(font)
        self.view3DSemi.setStyleSheet(buttonStyle)
        self.view3DSemi.setObjectName("view3DSemi")
        self.dicom_file = QtWidgets.QLabel(self.centralwidget)
        self.dicom_file.setGeometry(QtCore.QRect(600, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.dicom_file.setFont(font)
        self.dicom_file.setObjectName("dicom_file")
        self.generateVolumeAuto = QtWidgets.QPushButton(self.centralwidget)
        self.generateVolumeAuto.setEnabled(True)
        self.generateVolumeAuto.setGeometry(QtCore.QRect(1020, 90, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.generateVolumeAuto.setFont(font)
        self.generateVolumeAuto.setMouseTracking(False)
        self.generateVolumeAuto.setStyleSheet(generateVolumeStyle)
        self.generateVolumeAuto.setObjectName("generateVolumeAuto")
        self.automatic_intervention = QtWidgets.QGroupBox(self.centralwidget)
        self.automatic_intervention.setEnabled(True)
        self.automatic_intervention.setGeometry(QtCore.QRect(590, 180, 591, 111))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.automatic_intervention.setFont(font)
        self.automatic_intervention.setStyleSheet(groupBoxStyle)
        self.automatic_intervention.setTitle("")
        self.automatic_intervention.setObjectName("automatic_intervention")
        self.view3DAuto = QtWidgets.QPushButton(self.automatic_intervention)
        self.view3DAuto.setGeometry(QtCore.QRect(430, 60, 131, 28))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.view3DAuto.setFont(font)
        self.view3DAuto.setStyleSheet(buttonStyle)
        self.view3DAuto.setObjectName("view3DAuto")
        self.volume_detected = QtWidgets.QLabel(self.automatic_intervention)
        self.volume_detected.setGeometry(QtCore.QRect(30, 30, 171, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.volume_detected.setFont(font)
        self.volume_detected.setObjectName("volume_detected")
        self.volume = QtWidgets.QLineEdit(self.automatic_intervention)
        self.volume.setGeometry(QtCore.QRect(30, 55, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.volume.setFont(font)
        self.volume.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.volume.setText("")
        self.volume.setFrame(False)
        self.volume.setReadOnly(True)
        self.volume.setObjectName("volume")
        self.slicesManager = QtWidgets.QGroupBox(self.centralwidget)
        self.slicesManager.setEnabled(False)
        self.slicesManager.setGeometry(QtCore.QRect(30, 540, 521, 130))
        self.slicesManager.setStyleSheet(boxStyle)
        self.slicesManager.setTitle("")
        self.slicesManager.setObjectName("slicesManager")
        self.total_slices = QtWidgets.QTextEdit(self.slicesManager)
        self.total_slices.setGeometry(QtCore.QRect(300, 10, 41, 41))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.total_slices.setFont(font)
        self.total_slices.setMouseTracking(True)
        self.total_slices.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.total_slices.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.total_slices.setLineWidth(1)
        self.total_slices.setAutoFormatting(QtWidgets.QTextEdit.AutoNone)
        self.total_slices.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.total_slices.setPlaceholderText("")
        self.total_slices.setObjectName("total_slices")
        self.label_2 = QtWidgets.QLabel(self.slicesManager)
        self.label_2.setGeometry(QtCore.QRect(180, 10, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.of = QtWidgets.QLabel(self.slicesManager)
        self.of.setGeometry(QtCore.QRect(260, 10, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.of.setFont(font)
        self.of.setAlignment(QtCore.Qt.AlignCenter)
        self.of.setObjectName("of")
        self.previous = QtWidgets.QPushButton(self.slicesManager)
        self.previous.setGeometry(QtCore.QRect(10, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.previous.setFont(font)
        self.previous.setStyleSheet(buttonStyle)
        self.previous.setObjectName("previous")
        self.next = QtWidgets.QPushButton(self.slicesManager)
        self.next.setGeometry(QtCore.QRect(420, 10, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.next.setFont(font)
        self.next.setStyleSheet(buttonStyle)
        self.next.setObjectName("next")
        self.sliceSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.slicesManager)
        self.sliceSlider.setGeometry(QtCore.QRect(10, 50, 501, 22))
        self.sliceSlider.setMinimum(1)
        self.sliceSlider.setMaximum(1)
        self.sliceSlider.setValue(1)
        self.sliceSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.sliceSlider.setObjectName("sliceSlider")
        self.edit_slice = QtWidgets.QPushButton(self.slicesManager)
        self.edit_slice.setGeometry(QtCore.QRect(180, 90, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        self.edit_slice.setFont(font)
        self.edit_slice.setStyleSheet(buttonStyle)
        self.edit_slice.setObjectName("edit_slice")
        self.actual_slice = QtWidgets.QTextEdit(self.slicesManager)
        self.actual_slice.setGeometry(QtCore.QRect(232, 10, 41, 41))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.actual_slice.setFont(font)
        self.actual_slice.setMouseTracking(True)
        self.actual_slice.setStyleSheet("background-color: rgba(0,0,0,0%);")
        self.actual_slice.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.actual_slice.setLineWidth(1)
        self.actual_slice.setAutoFormatting(QtWidgets.QTextEdit.AutoNone)
        self.actual_slice.setReadOnly(True)
        self.actual_slice.setPlaceholderText("")
        self.actual_slice.setObjectName("actual_slice")
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setGeometry(QtCore.QRect(1020, 600, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.closeButton.setFont(font)
        self.closeButton.setStyleSheet(closeStyle)
        self.closeButton.setObjectName("closeButton")
        self.autoLabel = QtWidgets.QLabel(self.centralwidget)
        self.autoLabel.setGeometry(QtCore.QRect(610, 170, 181, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.autoLabel.setFont(font)
        self.autoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.autoLabel.setObjectName("autoLabel")
        self.semiLabel = QtWidgets.QLabel(self.centralwidget)
        self.semiLabel.setGeometry(QtCore.QRect(610, 310, 181, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        self.semiLabel.setFont(font)
        self.semiLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.semiLabel.setObjectName("semiLabel")
        self.infoAuto = QtWidgets.QGroupBox(self.centralwidget)
        self.infoAuto.setGeometry(QtCore.QRect(910, 140, 251, 41))
        self.infoAuto.setStyleSheet(boxStyle)
        self.infoAuto.setTitle("")
        self.infoAuto.setObjectName("infoAuto")
        self.textAuto = QtWidgets.QLabel(self.infoAuto)
        self.textAuto.setGeometry(QtCore.QRect(30, 8, 221, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(7)
        font.setBold(False)
        font.setWeight(50)
        self.textAuto.setFont(font)
        self.textAuto.setStyleSheet("background-color:none;")
        self.textAuto.setObjectName("textAuto")
        self.iconAuto = QtWidgets.QLabel(self.infoAuto)
        self.iconAuto.setGeometry(QtCore.QRect(10, 10, 15, 15))
        self.iconAuto.setAutoFillBackground(False)
        self.iconAuto.setText("")
        self.iconAuto.setPixmap(QtGui.QPixmap(":/aux/info.png"))
        self.iconAuto.setScaledContents(True)
        self.iconAuto.setObjectName("iconAuto")
        self.automatic_intervention.raise_()
        self.manual_intervention.raise_()
        self.filename.raise_()
        self.openFile.raise_()
        self.imageHeart.raise_()
        self.dicom_file.raise_()
        self.generateVolumeAuto.raise_()
        self.slicesManager.raise_()
        self.closeButton.raise_()
        self.autoLabel.raise_()
        self.semiLabel.raise_()
        self.infoAuto.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1208, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")

        'Flag to manager the text shown in slices label'
        self.newEdition = False

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    'WINDOWS MANAGER'

    def searchFolder(self):
        global dir
        dir = QFileDialog.getExistingDirectory()
        self.filename.setText(dir)
        self.enableGenerateVolume()

    def openWindow(self):
        self.clearEditLine()

        'Variables to send to edit window'
        self.patient_edit = patient_id
        current_slice = int(self.actual_slice.toPlainText())
        self.slice_edit = current_slice

        "Send data to second window"
        self.secondwindow = SecondWindow(self.slice_edit, self.patient_edit)
        self.secondwindow.submited.connect(self.updateImage)

    def closeProgram(self):
        self.cleanAll()
        QtWidgets.QApplication.quit()

    def cleanAll(self):
        base = 'aux_img'
        if not os.path.isdir(base):
            return
        for f in os.listdir(base):
            subdir = os.path.join(base, f)
            if not os.path.isdir(subdir):
                continue
            for subf in os.listdir(subdir):
                try:
                    os.remove(os.path.join(subdir, subf))
                except OSError:
                    pass

    def closeEvent(self, event):
        msgBox = QMessageBox()
        msgBox.setText("Close Program")
        msgBox.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes | QMessageBox.No);
        msgBox.setDefaultButton(QMessageBox.Yes)
        if msgBox.exec_() == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    'GENERATE AUTOMATIC AND SEMIAUTOMATIC VOLUME'

    def generateVolume(self):
        global patient_id
        global no_slices
        patient_id, no_slices, vol = _automatic_.segmentEpicardialFat(DICOM_DATASET=dir, OUTPUT_FOLDER='aux_img/')
        self.volume.setText(str(round(vol, 1)))
        self.refreshAfterAutomatic()

    def generateNewVolume(self):
        patient_id, no_slices, vol = _semiautomatic_.segmentEpicardialFat(DICOM_DATASET=dir, OUTPUT_FOLDER='aux_img/')
        self.volume_2.setText(str(round(vol, 1)))
        self.refreshAfterSemiAutomatic()

    'INTERFACE MANAGER'

    def updateImage(self, slice_id):
        "Show container"
        self.manual_intervention.show()
        self.semiLabel.show()
        self.afterEditionBox.hide()
        self.slice_id = slice_id
        self.editionHandler()

        'Show the image with new contour'
        self.imageHeart.setPixmap(QtGui.QPixmap(':/aux/edited_successfully_bg.png'))

    def clearEditLine(self):
        'Clean the slices label'
        if self.newEdition:
            self.slices.setText('')
            self.newEdition = False

    def refreshAfterAutomatic(self):
        self.infoAuto.hide()
        self.automatic_intervention.show()
        self.autoLabel.show()
        self.slicesManager.setEnabled(True)
        self.setFirstSlice()

    def refreshAfterSemiAutomatic(self):
        self.afterEditionBox.show()
        self.setFirstSlice()
        self.newEdition = True

    def enableGenerateVolume(self):
        if self.filename.text():
            self.generateVolumeAuto.show()
            self.infoAuto.show()

    'SLICE NAVIGATOR'
    def setFirstSlice(self):
        'Show the number of slices and the first slice'
        self.total_slices.setText(str(no_slices - 1))
        self.actual_slice.setText(str(1))
        self.sliceSlider.setMaximum(no_slices - 1)
        self.sliceSlider.setValue(1)
        'Show image of slice 0'
        self.imageHeart.setPixmap(QtGui.QPixmap(f"aux_img/combined/{patient_id}_{0}_combined.png"))

    def getCurrentSlice(self):
        return self.actual_slice.toPlainText()

    def displaySlice(self):
        current = self.getCurrentSlice()
        slice = int(current) - 1
        self.imageHeart.setPixmap(QtGui.QPixmap(f"aux_img/combined/{patient_id}_{slice}_combined.png"))

    def nextSlice(self):
        current = int(self.getCurrentSlice())
        new = min(current + 1, no_slices - 1)
        self.actual_slice.setText(str(new))
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setValue(new)
        self.sliceSlider.blockSignals(False)
        self.displaySlice()

    def previousSlice(self):
        current = int(self.getCurrentSlice())
        new = max(current - 1, 1)
        self.actual_slice.setText(str(new))
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setValue(new)
        self.sliceSlider.blockSignals(False)
        self.displaySlice()

    def sliderChanged(self, value):
        self.actual_slice.setText(str(value))
        self.displaySlice()

    def editionHandler(self):
        current = int(self.getCurrentSlice())
        current_text = self.slices.toPlainText()

        if not current_text:
            self.slices.setText(str(current))
        else:
            current_text = f'{current_text}, {current}'
            self.slices.setText(str(current_text))

    'RETRANSLATE UI'

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("HARTA - Epicardial Fat Segmentation Software", "HARTA - Epicardial Fat Segmentation Software"))
        MainWindow.setWindowIcon(QtGui.QIcon(':/aux/logo_w.png'))
        MainWindow.setIconSize(QtCore.QSize(500, 200))
        self.openFile.setText(_translate("MainWindow", "Open Exam..."))
        self.generateVolumeSemi.setText(_translate("MainWindow", "Calculate New \n"
                                                                 "Volume"))
        self.slices_edited.setText(_translate("MainWindow", "Slices edited:"))
        self.textAuto_2.setText(_translate("MainWindow", "This process may takes a few seconds."))
        self.new_volume.setText(_translate("MainWindow", "Epicardial Fat Volume (ml)"))
        self.view3DSemi.setText(_translate("MainWindow", "3D View"))
        self.dicom_file.setText(_translate("MainWindow", "DICOM File"))
        self.generateVolumeAuto.setText(_translate("MainWindow", "Calculate \n"
                                                                 "Volume"))
        self.view3DAuto.setText(_translate("MainWindow", "3D View"))
        self.volume_detected.setText(_translate("MainWindow", "Epicardial Fat Volume (ml)"))
        self.label_2.setText(_translate("MainWindow", "Slice"))
        self.of.setText(_translate("MainWindow", "of"))
        self.previous.setText(_translate("MainWindow", "« Previous"))
        self.next.setText(_translate("MainWindow", "Next »"))
        self.edit_slice.setText(_translate("MainWindow", "Edit slice"))
        self.closeButton.setText(_translate("MainWindow", "Close"))
        self.autoLabel.setText(_translate("MainWindow", "Automatic detection"))
        self.semiLabel.setText(_translate("MainWindow", "Manual intervention"))
        self.textAuto.setText(_translate("MainWindow", "This process may takes a few seconds."))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File..."))

        'ASSIGN FUNCTIONS'
        'Hide all elements'
        self.generateVolumeAuto.hide()
        self.automatic_intervention.hide()
        self.manual_intervention.hide()
        self.infoAuto.hide()
        self.autoLabel.hide()
        self.semiLabel.hide()

        self.view3DAuto.hide()
        self.view3DSemi.hide()

        'Button to search file'
        self.openFile.clicked.connect(self.searchFolder)

        'Button to generate volume'
        self.generateVolumeAuto.clicked.connect(self.generateVolume)
        self.generateVolumeSemi.clicked.connect(self.generateNewVolume)

        'Edit slice button'
        self.edit_slice.clicked.connect(self.openWindow)

        'Button to go to next and previous slice'
        self.next.clicked.connect(self.nextSlice)
        self.previous.clicked.connect(self.previousSlice)

        'Slider to navigate slices'
        self.sliceSlider.valueChanged.connect(self.sliderChanged)

        'Button to close the program'
        self.closeButton.clicked.connect(self.closeProgram)

        # Wire scroll-to-edit and double-click on slice image
        self.imageHeart.setNavigator(self)


'***************************************************************************************************************************************************'
'SECOND WINDOW'
'***************************************************************************************************************************************************'


class SecondWindow(QWidget):
    submited = QtCore.pyqtSignal(int)

    def __init__(self, slice_edit, patient_edit):
        super().__init__()
        self.slice_edit = slice_edit
        self.patient_edit = patient_edit
        self.eventhandler = EventHandler(self)
        self._orig = cv.imread(
            f"aux_img/slices/{patient_edit}_{slice_edit}.png",
            cv.IMREAD_GRAYSCALE
        )
        self._wl_center = 128
        self._wl_width  = 256
        self.setupUi()

    # ── Windowing helpers ────────────────────────────────────────────────────

    def _applyWindowing(self):
        lo = max(self._wl_center - self._wl_width // 2, 0)
        hi = min(self._wl_center + self._wl_width // 2, 255)
        if hi <= lo:
            hi = lo + 1
        scale = 255.0 / (hi - lo)
        win = np.clip((self._orig.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)
        rgb = cv.cvtColor(win, cv.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.img.setPixmap(QPixmap.fromImage(qimg))
        self.img.update()

    def _onCenterChanged(self, v):
        self._wl_center = v
        self._centerSpin.blockSignals(True);  self._centerSpin.setValue(v);  self._centerSpin.blockSignals(False)
        self._applyWindowing()

    def _onCenterSpinChanged(self, v):
        self._wl_center = v
        self._centerSlider.blockSignals(True); self._centerSlider.setValue(v); self._centerSlider.blockSignals(False)
        self._applyWindowing()

    def _onWidthChanged(self, v):
        self._wl_width = v
        self._widthSpin.blockSignals(True);  self._widthSpin.setValue(v);  self._widthSpin.blockSignals(False)
        self._applyWindowing()

    def _onWidthSpinChanged(self, v):
        self._wl_width = v
        self._widthSlider.blockSignals(True); self._widthSlider.setValue(v); self._widthSlider.blockSignals(False)
        self._applyWindowing()

    def _applyPreset(self, center, width):
        self._wl_center, self._wl_width = center, width
        for w in (self._centerSlider, self._centerSpin, self._widthSlider, self._widthSpin):
            w.blockSignals(True)
        self._centerSlider.setValue(center); self._centerSpin.setValue(center)
        self._widthSlider.setValue(width);   self._widthSpin.setValue(width)
        for w in (self._centerSlider, self._centerSpin, self._widthSlider, self._widthSpin):
            w.blockSignals(False)
        self._applyWindowing()

    def _setMode(self, mode):
        self.eventhandler.setMode(mode)
        activeStyle   = ("QPushButton { border-radius:8px; background-color:rgb(70,136,244); "
                         "color:white; border:none; } QPushButton:hover { background-color:rgb(18,77,150); }")
        inactiveStyle = ("QPushButton { border-radius:8px; background-color:rgb(234,234,234); "
                         "color:black; border:none; } QPushButton:hover { background-color:rgb(183,181,181); }")
        self._btnModeDraw.setStyleSheet(inactiveStyle)
        self._btnModeBrush.setStyleSheet(inactiveStyle)
        self._btnModeWindow.setStyleSheet(inactiveStyle)
        if mode == 'window':
            self._btnModeWindow.setStyleSheet(activeStyle)
            self._dragHint.show()
        elif mode == 'brush':
            self._btnModeBrush.setStyleSheet(activeStyle)
            self._dragHint.hide()
        else:  # 'draw'
            self._btnModeDraw.setStyleSheet(activeStyle)
            self._dragHint.hide()

    # ── Contour propagation to adjacent slices ───────────────────────────────

    def _propagateContour(self):
        n, ok = QInputDialog.getInt(
            self, "Propagate Contour",
            "Copy contour to ±N adjacent slices\n(slight morphological adaptation per step):",
            value=2, min=1, max=10
        )
        if not ok:
            return

        src_path = f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png'
        if not os.path.exists(src_path):
            QMessageBox.warning(self, "No contour saved",
                                "Click Done first to save the current contour,\nthen use Propagate.")
            return

        src_bgr  = cv.imread(src_path)
        src_gray = cv.cvtColor(src_bgr, cv.COLOR_BGR2GRAY)
        _, src_bin = cv.threshold(src_gray, 127, 255, cv.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        count  = 0

        for offset in range(-n, n + 1):
            if offset == 0:
                continue
            target = int(self.slice_edit) - 1 + offset
            if target < 0 or target >= no_slices - 1:
                continue

            iters  = abs(offset)
            # Erode forward (smaller), dilate backward (larger) to
            # account for the heart growing / shrinking across slices
            morphed = (cv.erode(src_bin, kernel, iterations=iters)
                       if offset > 0
                       else cv.dilate(src_bin, kernel, iterations=iters))

            # Save contour as 3-channel BGR (white on black) to match combineImages()
            morphed_bgr  = cv.cvtColor(morphed, cv.COLOR_GRAY2BGR)
            contour_path = f'aux_img/contours/{self.patient_edit}_{target}_c.png'
            cv.imwrite(contour_path, morphed_bgr)

            # Rebuild combined overlay
            slice_path = f'aux_img/slices/{self.patient_edit}_{target}.png'
            if os.path.exists(slice_path):
                overlay = morphed_bgr.copy()
                overlay[morphed > 0] = [22, 9, 224]
                img_slice = cv.imread(slice_path)
                if img_slice is not None:
                    out = cv.addWeighted(overlay, 0.2, img_slice, 1, 0, img_slice)
                    cv.imwrite(f'aux_img/combined/{self.patient_edit}_{target}_combined.png', out)
            count += 1

        QMessageBox.information(self, "Done",
                                f"Contour propagated to {count} adjacent slice(s).")

    # ── Hotkeys ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key  = event.key()
        mods = event.modifiers()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.onSubmit()
        elif key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_D:
            self._setMode('draw')
        elif key == Qt.Key_B:
            self._setMode('brush')
        elif key == Qt.Key_W:
            self._setMode('window')
        elif key == Qt.Key_Z and mods & Qt.ControlModifier:
            self.eventhandler.undoLastPoint()
        elif key == Qt.Key_1:
            self._applyPreset(128, 256)   # Full view
        elif key == Qt.Key_2:
            self._applyPreset(128, 150)   # Soft Tissue
        elif key == Qt.Key_3:
            self._applyPreset(210,  80)   # Bone
        elif key == Qt.Key_4:
            self._applyPreset( 64, 180)   # Lung
        else:
            super().keyPressEvent(event)

    # ── UI setup ─────────────────────────────────────────────────────────────

    def setupUi(self):
        self.setWindowTitle("Pericardium delineation")
        self.setStyleSheet("background-color:rgb(255, 255, 255);")
        self.setWindowIcon(QtGui.QIcon(':/aux/logo_w.png'))
        self.resize(1050, 750)
        self.setMinimumSize(QtCore.QSize(1050, 750))

        buttonStyle = """
        QPushButton{
            border-style: solid; border-width: 0px; border-radius: 10px;
            background-color: rgb(234,234,234);
        }
        QPushButton::hover{ background-color: rgb(183,181,181); }
        """
        presetStyle = """
        QPushButton{
            border-style: solid; border-width: 1px; border-color: rgb(200,200,200);
            border-radius: 8px; background-color: rgb(245,245,245);
        }
        QPushButton::hover{ background-color: rgb(70,136,244); color: white; border-color: rgb(70,136,244); }
        """

        fnt9 = QtGui.QFont(); fnt9.setFamily("Roboto"); fnt9.setPointSize(9)
        fntB = QtGui.QFont(); fntB.setFamily("Roboto"); fntB.setPointSize(10); fntB.setBold(True)
        fntS = QtGui.QFont(); fntS.setFamily("Roboto"); fntS.setPointSize(8)

        # ── Image (EventHandler) ─────────────────────────────────────────────
        self.img = self.eventhandler
        self._applyWindowing()
        self.img.setGeometry(QRect(20, 65, 512, 512))
        self.img.setMinimumSize(QtCore.QSize(512, 512))
        self.img.setScaledContents(True)
        self.img.setCursor(Qt.CrossCursor)

        # ── Mode selector ────────────────────────────────────────────────────
        PX = 560   # panel x origin

        modeActiveStyle = ("QPushButton { border-radius:8px; "
                           "background-color:rgb(70,136,244); color:white; border:none; }"
                           "QPushButton:hover { background-color:rgb(18,77,150); }")
        modeInactiveStyle = ("QPushButton { border-radius:8px; "
                             "background-color:rgb(234,234,234); color:black; border:none; }"
                             "QPushButton:hover { background-color:rgb(183,181,181); }")

        modeLbl = QLabel("Mode:", self)
        modeLbl.setGeometry(QRect(PX, 70, 50, 26))
        modeLbl.setFont(fnt9)

        self._btnModeDraw = QPushButton("✏  Draw ROI", self)
        self._btnModeDraw.setGeometry(QRect(PX + 55, 65, 110, 30))
        self._btnModeDraw.setFont(fnt9)
        self._btnModeDraw.setStyleSheet(modeActiveStyle)
        self._btnModeDraw.clicked.connect(lambda: self._setMode('draw'))

        # ── Smart Brush (Magnetic Lasso) ─────────────────────────────────────
        self._btnModeBrush = QPushButton("🖌  Smart Brush", self)
        self._btnModeBrush.setGeometry(QRect(PX + 173, 65, 125, 30))
        self._btnModeBrush.setFont(fnt9)
        self._btnModeBrush.setStyleSheet(modeInactiveStyle)
        self._btnModeBrush.clicked.connect(lambda: self._setMode('brush'))

        self._btnModeWindow = QPushButton("🖱  Windowing", self)
        self._btnModeWindow.setGeometry(QRect(PX + 306, 65, 130, 30))
        self._btnModeWindow.setFont(fnt9)
        self._btnModeWindow.setStyleSheet(modeInactiveStyle)
        self._btnModeWindow.clicked.connect(lambda: self._setMode('window'))

        self._dragHint = QLabel("Drag on image:  ←→ Width   ↑↓ Center", self)
        self._dragHint.setGeometry(QRect(PX + 55, 99, 350, 16))
        self._dragHint.setFont(fntS)
        self._dragHint.setStyleSheet("color: rgb(100,140,220);")
        self._dragHint.hide()

        sep0 = QLabel(self)
        sep0.setGeometry(QRect(PX, 122, 460, 2))
        sep0.setStyleSheet("background-color: rgb(220,220,220);")

        # ── Windowing controls ───────────────────────────────────────────────
        panelTitle = QLabel("Windowing / Gray-level Mapping", self)
        panelTitle.setGeometry(QRect(PX, 132, 460, 26))
        panelTitle.setFont(fntB)

        sep = QLabel(self)
        sep.setGeometry(QRect(PX, 161, 460, 2))
        sep.setStyleSheet("background-color: rgb(220,220,220);")

        # Window Center (Level)
        QLabel("Window Center (Level)", self).setGeometry(QRect(PX, 171, 220, 18))
        self._centerSlider = QSlider(Qt.Horizontal, self)
        self._centerSlider.setGeometry(QRect(PX, 193, 340, 22))
        self._centerSlider.setRange(0, 255); self._centerSlider.setValue(128)
        self._centerSpin = QSpinBox(self)
        self._centerSpin.setGeometry(QRect(PX + 350, 189, 72, 30))
        self._centerSpin.setRange(0, 255); self._centerSpin.setValue(128)
        self._centerSpin.setFont(fnt9)

        # Window Width
        QLabel("Window Width", self).setGeometry(QRect(PX, 233, 220, 18))
        self._widthSlider = QSlider(Qt.Horizontal, self)
        self._widthSlider.setGeometry(QRect(PX, 255, 340, 22))
        self._widthSlider.setRange(1, 256); self._widthSlider.setValue(256)
        self._widthSpin = QSpinBox(self)
        self._widthSpin.setGeometry(QRect(PX + 350, 251, 72, 30))
        self._widthSpin.setRange(1, 256); self._widthSpin.setValue(256)
        self._widthSpin.setFont(fnt9)

        # Presets
        presetsLbl = QLabel("Presets:", self)
        presetsLbl.setGeometry(QRect(PX, 295, 70, 18))
        presetsLbl.setFont(fnt9)

        presets = [
            ("Full view",   128, 256),
            ("Soft Tissue", 128, 150),
            ("Bone",        210,  80),
            ("Lung",         64, 180),
        ]
        for i, (label, c, w) in enumerate(presets):
            btn = QPushButton(label, self)
            btn.setGeometry(QRect(PX + i * 110, 317, 100, 28))
            btn.setFont(fnt9)
            btn.setStyleSheet(presetStyle)
            btn.clicked.connect(lambda _, c=c, w=w: self._applyPreset(c, w))

        sep2 = QLabel(self)
        sep2.setGeometry(QRect(PX, 358, 460, 2))
        sep2.setStyleSheet("background-color: rgb(220,220,220);")

        instrLbl = QLabel(
            "✏  Draw ROI: click to place points, delineate boundary.\n"
            "🖌  Smart Brush: click and drag freehand — auto-snaps to\n"
            "     edges on release (Canny magnetic lasso).\n"
            "🖱  Windowing: drag on image — ← → Width, ↑ ↓ Center.", self)
        instrLbl.setGeometry(QRect(PX, 368, 440, 80))
        instrLbl.setFont(fntS)
        instrLbl.setStyleSheet("color: rgb(120,120,120);")

        # ── Contour tools section ────────────────────────────────────────────
        sep3 = QLabel(self)
        sep3.setGeometry(QRect(PX, 458, 460, 2))
        sep3.setStyleSheet("background-color: rgb(220,220,220);")

        toolsTitle = QLabel("Contour Tools", self)
        toolsTitle.setGeometry(QRect(PX, 468, 460, 22))
        toolsTitle.setFont(fntB)

        propagateBtn = QPushButton("↗  Propagate ±N slices", self)
        propagateBtn.setGeometry(QRect(PX, 496, 200, 30))
        propagateBtn.setFont(fnt9)
        propagateBtn.setStyleSheet(buttonStyle)
        propagateBtn.clicked.connect(self._propagateContour)

        propagateHint = QLabel("Copies the saved contour to adjacent slices\nwith slight morphological adaptation.", self)
        propagateHint.setGeometry(QRect(PX + 210, 496, 240, 32))
        propagateHint.setFont(fntS)
        propagateHint.setStyleSheet("color: rgb(120,120,120);")

        # ── Hotkeys reference ────────────────────────────────────────────────
        sep4 = QLabel(self)
        sep4.setGeometry(QRect(PX, 540, 460, 2))
        sep4.setStyleSheet("background-color: rgb(220,220,220);")

        hotkeysTitle = QLabel("Keyboard Shortcuts", self)
        hotkeysTitle.setGeometry(QRect(PX, 550, 460, 22))
        hotkeysTitle.setFont(fntB)

        hotkeysLbl = QLabel(
            "D  — Draw ROI mode          B  — Smart Brush mode\n"
            "W  — Windowing mode       1-4  — Apply preset\n"
            "Ctrl+Z  — Undo last point    Enter  — Done\n"
            "Esc     — Cancel", self)
        hotkeysLbl.setGeometry(QRect(PX, 576, 460, 68))
        hotkeysLbl.setFont(fntS)
        hotkeysLbl.setStyleSheet("color: rgb(80,80,80);")

        # Connect sliders ↔ spinboxes
        self._centerSlider.valueChanged.connect(self._onCenterChanged)
        self._centerSpin.valueChanged.connect(self._onCenterSpinChanged)
        self._widthSlider.valueChanged.connect(self._onWidthChanged)
        self._widthSpin.valueChanged.connect(self._onWidthSpinChanged)

        # ── Done / Cancel ────────────────────────────────────────────────────
        self.doneButton = QPushButton('Done', self)
        self.doneButton.setGeometry(QRect(820, 700, 100, 32))
        self.doneButton.setFont(fnt9)
        self.doneButton.setStyleSheet(buttonStyle)
        self.doneButton.clicked.connect(self.onSubmit)

        self.cancelButton = QPushButton('Cancel', self)
        self.cancelButton.setGeometry(QRect(930, 700, 100, 32))
        self.cancelButton.setFont(fnt9)
        self.cancelButton.setStyleSheet(buttonStyle)
        self.cancelButton.clicked.connect(lambda: self.close())

        self.show()

    'SUBMIT HANDLER'

    def onSubmit(self):
        'Get the drawing of path'
        path = self.eventhandler.getPath()
        self.image = QImage(800, 800, QImage.Format_RGB32)
        self.image.fill(Qt.black)
        painter = QPainter(self.image)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        painter.setPen(Qt.white)
        painter.fillPath(path, QBrush(QColor("white")))
        painter.drawPath(path)

        self.submited.emit(self.slice_edit)
        'Save path'
        filePath = f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png'
        self.image.save(filePath)
        'Rescale path to original size of dicom images'
        self.rescaleImage()
        'Save edited image'
        self.combineImages()
        'Clear path'
        self.eventhandler.clearPath()
        'Close the second window'
        self.close()

    def rescaleImage(self):
        filePath = f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png'
        img = cv.imread(filePath)
        dim = (512, 512)
        resized = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
        cv.imwrite(filePath, resized)

    def combineImages(self):
        mask_rgb = cv.imread(f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png')
        img = cv.imread(f'aux_img/slices/{self.patient_edit}_{int(self.slice_edit) - 1}.png')
        mask_rgb[np.where((mask_rgb == [255, 255, 255]).all(axis=2))] = [22, 9, 224]
        out = cv.addWeighted(mask_rgb, 0.2, img, 1, 0, img)
        filepath = f'aux_img/combined/{self.patient_edit}_{int(self.slice_edit) - 1}_combined.png'
        cv.imwrite(filepath, out)


class EventHandler(QLabel):
    """Image canvas supporting three interaction modes:
       'draw'   – click to place ROI contour points (default)
       'brush'  – drag freehand; on release auto-snaps to Canny edges
       'window' – click-drag to adjust windowing (dx→Width, dy→Center)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flag   = True
        self.points = []
        self.path   = QtGui.QPainterPath()
        self._mode        = 'draw'
        self._drag_start  = None
        self._drag_center = 128
        self._drag_width  = 256
        self._brushing    = False

    def setMode(self, mode):
        self._mode = mode
        cursors = {'draw': Qt.CrossCursor, 'brush': Qt.PointingHandCursor, 'window': Qt.SizeAllCursor}
        self.setCursor(cursors.get(mode, Qt.CrossCursor))

    # ── Mouse events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._mode == 'draw':
            self.points.append(event.pos())
            self.update()
        elif self._mode == 'brush':
            self._brushing = True
            self.points    = []
            self.points.append(event.pos())
            self.update()
        else:                               # windowing mode
            self._drag_start  = event.pos()
            sw = self.parent()
            if hasattr(sw, '_wl_center'):
                self._drag_center = sw._wl_center
                self._drag_width  = sw._wl_width

    def mouseMoveEvent(self, event):
        if self._mode == 'brush' and self._brushing:
            self.points.append(event.pos())
            self.update()
        elif self._mode == 'window' and self._drag_start is not None:
            dx = event.pos().x() - self._drag_start.x()
            dy = event.pos().y() - self._drag_start.y()
            new_center = max(0,   min(255, int(self._drag_center - dy)))
            new_width  = max(1,   min(256, int(self._drag_width  + dx)))
            sw = self.parent()
            if hasattr(sw, '_applyPreset'):
                sw._applyPreset(new_center, new_width)

    def mouseReleaseEvent(self, event):
        if self._mode == 'brush' and self._brushing:
            self._brushing = False
            self._simplifyPoints()
            self._snapToEdges()
            self.update()
        self._drag_start = None

    # ── Smart Brush helpers ──────────────────────────────────────────────────

    def _simplifyPoints(self, epsilon=3.0):
        """Ramer-Douglas-Peucker simplification to reduce freehand point count."""
        if len(self.points) < 3:
            return
        pts = np.array([[p.x(), p.y()] for p in self.points],
                       dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv.approxPolyDP(pts, epsilon, closed=True)
        self.points = [QtCore.QPoint(int(p[0][0]), int(p[0][1])) for p in simplified]

    def _snapToEdges(self, radius=20):
        """Magnetic-lasso: snap each point to the nearest Canny edge pixel."""
        parent = self.parent()
        if not hasattr(parent, '_orig') or parent._orig is None:
            return

        orig   = parent._orig           # grayscale uint8
        img_h, img_w = orig.shape
        w_w, w_h     = self.width(), self.height()

        edges = cv.Canny(orig, 30, 100)
        # Dilate so snapping is forgiving even if the cursor missed by a few pixels
        kernel = np.ones((5, 5), np.uint8)
        edges  = cv.dilate(edges, kernel)

        snapped = []
        for p in self.points:
            # widget coords → image coords
            ix = int(p.x() * img_w / w_w)
            iy = int(p.y() * img_h / w_h)
            ix = max(0, min(img_w - 1, ix))
            iy = max(0, min(img_h - 1, iy))

            x1 = max(0, ix - radius); x2 = min(img_w, ix + radius + 1)
            y1 = max(0, iy - radius); y2 = min(img_h, iy + radius + 1)
            region   = edges[y1:y2, x1:x2]
            edge_yx  = np.argwhere(region > 0)

            if len(edge_yx) > 0:
                dists = np.hypot(edge_yx[:, 1] - (ix - x1),
                                 edge_yx[:, 0] - (iy - y1))
                best = edge_yx[np.argmin(dists)]
                nx = int((x1 + best[1]) * w_w / img_w)
                ny = int((y1 + best[0]) * w_h / img_h)
                snapped.append(QtCore.QPoint(nx, ny))
            else:
                snapped.append(p)
        self.points = snapped

    def undoLastPoint(self):
        """Remove the most recently placed point (works in draw and brush modes)."""
        if self.points:
            self.points.pop()
            self.update()

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.flag:
            self.points = []
            self.path   = QtGui.QPainterPath()
            self.flag   = False
        painter = QPainter(self)
        painter.setPen(QPen(QColor(224, 9, 22), 10, Qt.SolidLine))
        for pos in self.points:
            painter.drawPoint(pos)
        if len(self.points) > 2:
            self.buildPath()
            painter.setPen(QPen(QColor(70, 136, 244), 3, Qt.SolidLine))
            painter.drawPath(self.path)

    # ── Path handler ────────────────────────────────────────────────────────

    def buildPath(self):
        factor = 0.5
        self.path = QtGui.QPainterPath(self.points[0])
        for p, current in enumerate(self.points[1:-1], 1):
            source      = QtCore.QLineF(self.points[p - 1], current)
            target      = QtCore.QLineF(current, self.points[p + 1])
            targetAngle = target.angleTo(source)
            if targetAngle > 180:
                angle = (source.angle() + source.angleTo(target) / 2) % 360
            else:
                angle = (target.angle() + target.angleTo(source) / 2) % 360
            revTarget = QtCore.QLineF.fromPolar(source.length() * factor, angle + 180).translated(current)
            cp2 = revTarget.p2()
            if p == 1:
                self.path.quadTo(cp2, current)
            else:
                self.path.cubicTo(cp1, cp2, current)
            revSource = QtCore.QLineF.fromPolar(target.length() * factor, angle).translated(current)
            cp1 = revSource.p2()
        self.path.quadTo(cp1, self.points[-1])

    def getPath(self):
        return self.path

    def clearPath(self):
        self.flag = True


def _safe_excepthook(exc_type, exc_value, exc_tb):
    import traceback
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(msg, file=sys.stderr)
    box = QMessageBox()
    box.setWindowTitle("Error")
    box.setText(f"{exc_type.__name__}: {exc_value}")
    box.setDetailedText(msg)
    box.exec_()

if __name__ == "__main__":
    # Ensure required directories exist
    for d in ['aux_img/combined', 'aux_img/slices', 'aux_img/contours', 'aux_img/fat']:
        os.makedirs(d, exist_ok=True)

    sys.excepthook = _safe_excepthook

    app = QtWidgets.QApplication(sys.argv)

    # Prevent PyQt5 5.15 from calling abort() on unhandled slot exceptions
    import traceback
    def _qt_exception_hook(exc_type, exc_value, exc_tb):
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.stderr)
        box = QMessageBox()
        box.setWindowTitle("Error")
        box.setText(f"{exc_type.__name__}: {exc_value}")
        box.setDetailedText(msg)
        box.exec_()
    sys.excepthook = _qt_exception_hook

    MainWindow = MainWindowClass()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setUi(ui)
    MainWindow.show()
    sys.exit(app.exec_())
