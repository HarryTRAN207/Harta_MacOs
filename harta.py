from PyQt5.QtWidgets import (
    QLabel, QWidget, QPushButton, QFileDialog, QMessageBox,
    QSlider, QSpinBox, QInputDialog, QProgressDialog
)
from PyQt5.QtGui import QBrush, QPen, QPainter, QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import os
import cv2 as cv
import numpy as np

import _automatic_
import _semiautomatic_
import resources

global flag


# ══════════════════════════════════════════════════════════════════════════════
# QThread worker — keeps UI responsive during heavy segmentation
# ══════════════════════════════════════════════════════════════════════════════
class SegmentationWorker(QThread):
    finished = pyqtSignal(str, int, float)   # patient_id, no_slices, vol
    failed   = pyqtSignal(str)

    def __init__(self, mode, dicom_dir, output_folder):
        super().__init__()
        self.mode          = mode           # 'auto' | 'semi'
        self.dicom_dir     = dicom_dir
        self.output_folder = output_folder

    def run(self):
        try:
            if self.mode == 'auto':
                pid, ns, vol = _automatic_.segmentEpicardialFat(
                    DICOM_DATASET=self.dicom_dir, OUTPUT_FOLDER=self.output_folder)
            else:
                pid, ns, vol = _semiautomatic_.segmentEpicardialFat(
                    DICOM_DATASET=self.dicom_dir, OUTPUT_FOLDER=self.output_folder)
            self.finished.emit(pid, ns, vol)
        except Exception as e:
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Scroll-to-Edit label: wheel scrolls slices, double-click opens editor
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# Main window with hotkey support
# ══════════════════════════════════════════════════════════════════════════════
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
                ui.nextSlice(); return
            elif key in (Qt.Key_Left, Qt.Key_Up):
                ui.previousSlice(); return
            elif key == Qt.Key_E:
                ui.openWindow(); return
        super().keyPressEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW UI
# ══════════════════════════════════════════════════════════════════════════════
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        buttonStyle = """
        QPushButton{border-style:solid;border-width:0px;border-radius:10px;
                    background-color:rgb(234,234,234);}
        QPushButton::hover{background-color:rgb(183,181,181);}"""
        groupBoxStyle = """
        QGroupBox{background-color:none;border:1px solid rgb(183,181,181);border-radius:10px;}"""
        closeStyle = """
        QPushButton{border-style:solid;border-width:0px;border-radius:10px;
                    background-color:rgb(234,67,53);color:rgb(255,255,255);}
        QPushButton::hover{background-color:rgb(172,16,12);}"""
        generateVolumeStyle = """
        QPushButton{border-style:solid;border-width:0px;border-radius:10px;
                    background-color:rgb(70,136,244);color:rgb(255,255,255);}
        QPushButton::hover{background-color:rgb(18,77,150);}"""
        boxStyle = "QGroupBox{border-style:none;background-color:none;}"
        textEditStyle = """
        QTextEdit{border-style:solid;border-width:1px;border-radius:10px;
                  border-color:rgb(183,181,181);}"""
        lineEditStyle = """
        QLineEdit{border-style:solid;border-width:1px;border-radius:10px;
                  border-color:rgb(183,181,181);}"""

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1208, 700)
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sp.setHorizontalStretch(0); sp.setVerticalStretch(0)
        MainWindow.setSizePolicy(sp)
        MainWindow.setMinimumSize(QtCore.QSize(1208, 700))
        MainWindow.setMaximumSize(QtCore.QSize(1208, 700))
        MainWindow.setStyleSheet("background-color:rgb(255,255,255);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.filename = QtWidgets.QLineEdit(self.centralwidget)
        self.filename.setGeometry(QtCore.QRect(740, 50, 411, 28))
        f = QtGui.QFont(); f.setFamily("Roboto")
        self.filename.setFont(f)
        self.filename.setStyleSheet(lineEditStyle)
        self.filename.setFrame(False)
        self.filename.setObjectName("filename")

        self.openFile = QtWidgets.QPushButton(self.centralwidget)
        self.openFile.setGeometry(QtCore.QRect(600, 50, 131, 28))
        fb = QtGui.QFont(); fb.setFamily("Roboto"); fb.setPointSize(9)
        fb.setBold(True); fb.setWeight(75)
        self.openFile.setFont(fb)
        self.openFile.setStyleSheet(buttonStyle)
        self.openFile.setObjectName("openFile")

        # Slice view (ClickableLabel)
        self.imageHeart = ClickableLabel(self.centralwidget)
        self.imageHeart.setGeometry(QtCore.QRect(30, 20, 520, 520))
        self.imageHeart.setText("")
        self.imageHeart.setPixmap(QtGui.QPixmap(":/aux/default_txt.png"))
        self.imageHeart.setScaledContents(True)
        self.imageHeart.setAlignment(QtCore.Qt.AlignCenter)
        self.imageHeart.setObjectName("imageHeart")

        self.manual_intervention = QtWidgets.QGroupBox(self.centralwidget)
        self.manual_intervention.setGeometry(QtCore.QRect(590, 320, 591, 211))
        fm = QtGui.QFont(); fm.setFamily("Roboto"); fm.setPointSize(10)
        self.manual_intervention.setFont(fm)
        self.manual_intervention.setStyleSheet(groupBoxStyle)
        self.manual_intervention.setTitle("")
        self.manual_intervention.setObjectName("manual_intervention")

        self.duringEditionBox = QtWidgets.QGroupBox(self.manual_intervention)
        self.duringEditionBox.setGeometry(QtCore.QRect(0, 15, 571, 91))
        fd = QtGui.QFont(); fd.setFamily("Roboto")
        self.duringEditionBox.setFont(fd)
        self.duringEditionBox.setStyleSheet(boxStyle)
        self.duringEditionBox.setTitle("")
        self.duringEditionBox.setObjectName("duringEditionBox")

        self.generateVolumeSemi = QtWidgets.QPushButton(self.duringEditionBox)
        self.generateVolumeSemi.setGeometry(QtCore.QRect(430, 0, 131, 51))
        fgs = QtGui.QFont(); fgs.setFamily("Roboto"); fgs.setPointSize(9)
        fgs.setBold(True); fgs.setWeight(75)
        self.generateVolumeSemi.setFont(fgs)
        self.generateVolumeSemi.setStyleSheet(generateVolumeStyle)
        self.generateVolumeSemi.setObjectName("generateVolumeSemi")

        self.slices = QtWidgets.QTextEdit(self.duringEditionBox)
        self.slices.setGeometry(QtCore.QRect(130, 3, 281, 31))
        fsl = QtGui.QFont(); fsl.setPointSize(9)
        self.slices.setFont(fsl)
        self.slices.setStyleSheet(textEditStyle)
        self.slices.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.slices.setReadOnly(True)
        self.slices.setObjectName("slices")

        self.slices_edited = QtWidgets.QLabel(self.duringEditionBox)
        self.slices_edited.setGeometry(QtCore.QRect(30, 10, 91, 16))
        fse = QtGui.QFont(); fse.setFamily("Roboto"); fse.setPointSize(9)
        self.slices_edited.setFont(fse)
        self.slices_edited.setObjectName("slices_edited")

        self.infoAuto_2 = QtWidgets.QGroupBox(self.duringEditionBox)
        self.infoAuto_2.setGeometry(QtCore.QRect(320, 50, 251, 41))
        self.infoAuto_2.setStyleSheet(boxStyle)
        self.infoAuto_2.setTitle("")
        self.infoAuto_2.setObjectName("infoAuto_2")
        self.textAuto_2 = QtWidgets.QLabel(self.infoAuto_2)
        self.textAuto_2.setGeometry(QtCore.QRect(30, 10, 211, 21))
        ft2 = QtGui.QFont(); ft2.setFamily("Roboto"); ft2.setPointSize(7)
        self.textAuto_2.setFont(ft2)
        self.textAuto_2.setStyleSheet("background-color:none;")
        self.textAuto_2.setObjectName("textAuto_2")
        self.iconAuto_2 = QtWidgets.QLabel(self.infoAuto_2)
        self.iconAuto_2.setGeometry(QtCore.QRect(10, 10, 15, 15))
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
        fv2 = QtGui.QFont(); fv2.setPointSize(15)
        self.volume_2.setFont(fv2)
        self.volume_2.setStyleSheet("background-color:rgba(0,0,0,0%);")
        self.volume_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.volume_2.setReadOnly(True)
        self.volume_2.setObjectName("volume_2")

        self.new_volume = QtWidgets.QLabel(self.afterEditionBox)
        self.new_volume.setGeometry(QtCore.QRect(30, 20, 180, 16))
        fnv = QtGui.QFont(); fnv.setFamily("Roboto"); fnv.setPointSize(9)
        self.new_volume.setFont(fnv)
        self.new_volume.setObjectName("new_volume")

        self.view3DSemi = QtWidgets.QPushButton(self.afterEditionBox)
        self.view3DSemi.setGeometry(QtCore.QRect(430, 60, 131, 28))
        fv3s = QtGui.QFont(); fv3s.setFamily("Roboto"); fv3s.setPointSize(9)
        self.view3DSemi.setFont(fv3s)
        self.view3DSemi.setStyleSheet(buttonStyle)
        self.view3DSemi.setObjectName("view3DSemi")

        self.dicom_file = QtWidgets.QLabel(self.centralwidget)
        self.dicom_file.setGeometry(QtCore.QRect(600, 20, 120, 16))
        fdf = QtGui.QFont(); fdf.setFamily("Roboto"); fdf.setPointSize(9)
        self.dicom_file.setFont(fdf)
        self.dicom_file.setObjectName("dicom_file")

        self.generateVolumeAuto = QtWidgets.QPushButton(self.centralwidget)
        self.generateVolumeAuto.setGeometry(QtCore.QRect(1020, 90, 131, 51))
        fgva = QtGui.QFont(); fgva.setFamily("Roboto"); fgva.setPointSize(9)
        fgva.setBold(True); fgva.setWeight(75)
        self.generateVolumeAuto.setFont(fgva)
        self.generateVolumeAuto.setStyleSheet(generateVolumeStyle)
        self.generateVolumeAuto.setObjectName("generateVolumeAuto")

        self.automatic_intervention = QtWidgets.QGroupBox(self.centralwidget)
        self.automatic_intervention.setGeometry(QtCore.QRect(590, 180, 591, 111))
        fai = QtGui.QFont(); fai.setFamily("Roboto"); fai.setPointSize(10)
        self.automatic_intervention.setFont(fai)
        self.automatic_intervention.setStyleSheet(groupBoxStyle)
        self.automatic_intervention.setTitle("")
        self.automatic_intervention.setObjectName("automatic_intervention")

        self.view3DAuto = QtWidgets.QPushButton(self.automatic_intervention)
        self.view3DAuto.setGeometry(QtCore.QRect(430, 60, 131, 28))
        fv3a = QtGui.QFont(); fv3a.setPointSize(9)
        self.view3DAuto.setFont(fv3a)
        self.view3DAuto.setStyleSheet(buttonStyle)
        self.view3DAuto.setObjectName("view3DAuto")

        self.volume_detected = QtWidgets.QLabel(self.automatic_intervention)
        self.volume_detected.setGeometry(QtCore.QRect(30, 30, 171, 16))
        fvd = QtGui.QFont(); fvd.setPointSize(9)
        self.volume_detected.setFont(fvd)
        self.volume_detected.setObjectName("volume_detected")

        self.volume = QtWidgets.QLineEdit(self.automatic_intervention)
        self.volume.setGeometry(QtCore.QRect(30, 55, 131, 31))
        fv = QtGui.QFont(); fv.setPointSize(15)
        self.volume.setFont(fv)
        self.volume.setStyleSheet("background-color:rgba(0,0,0,0%);")
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
        fts = QtGui.QFont(); fts.setFamily("Roboto"); fts.setPointSize(11)
        self.total_slices.setFont(fts)
        self.total_slices.setStyleSheet("background-color:rgba(0,0,0,0%);")
        self.total_slices.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.total_slices.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.total_slices.setObjectName("total_slices")

        self.label_2 = QtWidgets.QLabel(self.slicesManager)
        self.label_2.setGeometry(QtCore.QRect(180, 10, 51, 31))
        fl2 = QtGui.QFont(); fl2.setFamily("Roboto"); fl2.setPointSize(9)
        self.label_2.setFont(fl2)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.of = QtWidgets.QLabel(self.slicesManager)
        self.of.setGeometry(QtCore.QRect(260, 10, 31, 31))
        fo = QtGui.QFont(); fo.setFamily("Roboto"); fo.setPointSize(9)
        self.of.setFont(fo)
        self.of.setAlignment(QtCore.Qt.AlignCenter)
        self.of.setObjectName("of")

        self.previous = QtWidgets.QPushButton(self.slicesManager)
        self.previous.setGeometry(QtCore.QRect(10, 10, 91, 31))
        fp = QtGui.QFont(); fp.setFamily("Roboto"); fp.setPointSize(9)
        self.previous.setFont(fp)
        self.previous.setStyleSheet(buttonStyle)
        self.previous.setToolTip("Lát cắt trước  (←)")
        self.previous.setObjectName("previous")

        self.next = QtWidgets.QPushButton(self.slicesManager)
        self.next.setGeometry(QtCore.QRect(420, 10, 91, 31))
        fn = QtGui.QFont(); fn.setFamily("Roboto"); fn.setPointSize(9)
        self.next.setFont(fn)
        self.next.setStyleSheet(buttonStyle)
        self.next.setToolTip("Lát cắt tiếp theo  (→)")
        self.next.setObjectName("next")

        self.sliceSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.slicesManager)
        self.sliceSlider.setGeometry(QtCore.QRect(10, 50, 501, 22))
        self.sliceSlider.setMinimum(1); self.sliceSlider.setMaximum(1)
        self.sliceSlider.setValue(1)
        self.sliceSlider.setStyleSheet("""
            QSlider::groove:horizontal{height:6px;background:rgb(220,220,220);border-radius:3px;}
            QSlider::handle:horizontal{width:16px;height:16px;background:rgb(70,136,244);
                border-radius:8px;margin:-5px 0;}
            QSlider::sub-page:horizontal{background:rgb(70,136,244);border-radius:3px;}""")
        self.sliceSlider.setObjectName("sliceSlider")

        self.edit_slice = QtWidgets.QPushButton(self.slicesManager)
        self.edit_slice.setGeometry(QtCore.QRect(180, 90, 161, 31))
        fes = QtGui.QFont(); fes.setFamily("Roboto"); fes.setPointSize(9)
        self.edit_slice.setFont(fes)
        self.edit_slice.setStyleSheet(buttonStyle)
        self.edit_slice.setToolTip("Mở cửa sổ chỉnh sửa contour  (E)")
        self.edit_slice.setObjectName("edit_slice")

        # Slice number — QSpinBox (replaces read-only QTextEdit, allows jump-to-slice)
        self.actual_slice = QtWidgets.QSpinBox(self.slicesManager)
        self.actual_slice.setGeometry(QtCore.QRect(228, 12, 52, 36))
        fas = QtGui.QFont(); fas.setFamily("Roboto"); fas.setPointSize(11)
        self.actual_slice.setFont(fas)
        self.actual_slice.setStyleSheet(
            "QSpinBox{background:transparent;border:none;}"
            "QSpinBox::up-button,QSpinBox::down-button{width:0;height:0;border:none;}")
        self.actual_slice.setRange(1, 1)
        self.actual_slice.setObjectName("actual_slice")

        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setGeometry(QtCore.QRect(1020, 600, 131, 31))
        fcb = QtGui.QFont(); fcb.setFamily("Roboto"); fcb.setPointSize(9)
        fcb.setBold(True); fcb.setWeight(75)
        self.closeButton.setFont(fcb)
        self.closeButton.setStyleSheet(closeStyle)
        self.closeButton.setObjectName("closeButton")

        self.autoLabel = QtWidgets.QLabel(self.centralwidget)
        self.autoLabel.setGeometry(QtCore.QRect(610, 170, 181, 16))
        fal = QtGui.QFont(); fal.setFamily("Roboto"); fal.setPointSize(11)
        self.autoLabel.setFont(fal)
        self.autoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.autoLabel.setObjectName("autoLabel")

        self.semiLabel = QtWidgets.QLabel(self.centralwidget)
        self.semiLabel.setGeometry(QtCore.QRect(610, 310, 181, 16))
        fsl2 = QtGui.QFont(); fsl2.setFamily("Roboto"); fsl2.setPointSize(11)
        self.semiLabel.setFont(fsl2)
        self.semiLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.semiLabel.setObjectName("semiLabel")

        self.infoAuto = QtWidgets.QGroupBox(self.centralwidget)
        self.infoAuto.setGeometry(QtCore.QRect(910, 140, 251, 41))
        self.infoAuto.setStyleSheet(boxStyle)
        self.infoAuto.setTitle("")
        self.infoAuto.setObjectName("infoAuto")
        self.textAuto = QtWidgets.QLabel(self.infoAuto)
        self.textAuto.setGeometry(QtCore.QRect(30, 8, 221, 21))
        fta = QtGui.QFont(); fta.setFamily("Roboto"); fta.setPointSize(7)
        self.textAuto.setFont(fta)
        self.textAuto.setStyleSheet("background-color:none;")
        self.textAuto.setObjectName("textAuto")
        self.iconAuto = QtWidgets.QLabel(self.infoAuto)
        self.iconAuto.setGeometry(QtCore.QRect(10, 10, 15, 15))
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

        # Slice display cache (avoids re-reading PNG from disk on every nav)
        self._slice_cache = {}

        self.newEdition = False
        self._worker = None  # holds reference to QThread while running

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ── Windows manager ──────────────────────────────────────────────────────

    def searchFolder(self):
        global dir
        dir = QFileDialog.getExistingDirectory()
        self.filename.setText(dir)
        self.enableGenerateVolume()

    def openWindow(self):
        self.clearEditLine()
        self.patient_edit = patient_id
        current_slice = self.actual_slice.value()
        self.slice_edit = current_slice
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
        msgBox.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.Yes)
        if msgBox.exec_() == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # ── Segmentation (QThread — non-blocking) ────────────────────────────────

    def generateVolume(self):
        self._startSegmentation('auto')

    def generateNewVolume(self):
        self._startSegmentation('semi')

    def _startSegmentation(self, mode):
        """Launch segmentation on a worker thread; show spinner; disable UI."""
        self._progress = QProgressDialog(
            "Đang phân đoạn mỡ ngoại tâm mạc…", "Huỷ", 0, 0,
            self.centralwidget.parent())
        self._progress.setWindowTitle("HARTA — Đang xử lý")
        self._progress.setWindowModality(Qt.ApplicationModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()
        QtWidgets.QApplication.processEvents()

        self.generateVolumeAuto.setEnabled(False)
        self.generateVolumeSemi.setEnabled(False)

        self._worker = SegmentationWorker(mode, dir, 'aux_img/')
        self._worker.finished.connect(self._onSegmentationDone)
        self._worker.failed.connect(self._onSegmentationFailed)
        self._progress.canceled.connect(self._worker.terminate)
        self._worker.start()

    def _onSegmentationDone(self, pid, ns, vol):
        global patient_id, no_slices
        patient_id, no_slices = pid, ns
        self._progress.close()
        self.generateVolumeAuto.setEnabled(True)
        self.generateVolumeSemi.setEnabled(True)
        self._slice_cache.clear()

        if self._worker and self._worker.mode == 'auto':
            self.volume.setText(str(round(vol, 1)))
            self.refreshAfterAutomatic()
        else:
            self.volume_2.setText(str(round(vol, 1)))
            self.refreshAfterSemiAutomatic()
        self.statusbar.showMessage(
            f"Phân đoạn hoàn tất — {ns} lát cắt  |  Volume: {round(vol,1)} ml", 5000)

    def _onSegmentationFailed(self, msg):
        self._progress.close()
        self.generateVolumeAuto.setEnabled(True)
        self.generateVolumeSemi.setEnabled(True)
        QMessageBox.critical(self.centralwidget.parent(), "Lỗi phân đoạn", msg)

    # ── Interface manager ────────────────────────────────────────────────────

    def updateImage(self, slice_id):
        self.manual_intervention.show()
        self.semiLabel.show()
        self.afterEditionBox.hide()
        self.slice_id = slice_id
        self.editionHandler()
        # Invalidate cache for edited slice
        self._slice_cache.pop(slice_id - 1, None)
        self._slice_cache.pop(slice_id - 2, None)
        self.displaySlice()
        self.statusbar.showMessage(f"Slice {slice_id} đã được cập nhật", 3000)

    def clearEditLine(self):
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

    # ── Slice navigator ──────────────────────────────────────────────────────

    def setFirstSlice(self):
        self._slice_cache.clear()
        self.total_slices.setText(str(no_slices - 1))
        self.actual_slice.blockSignals(True)
        self.actual_slice.setRange(1, no_slices - 1)
        self.actual_slice.setValue(1)
        self.actual_slice.blockSignals(False)
        self.sliceSlider.setMaximum(no_slices - 1)
        self.sliceSlider.setValue(1)
        self.imageHeart.setPixmap(QtGui.QPixmap(
            f"aux_img/combined/{patient_id}_{0}_combined.png"))

    def getCurrentSlice(self):
        return self.actual_slice.value()

    def displaySlice(self):
        idx = self.getCurrentSlice() - 1
        if idx not in self._slice_cache:
            path = f"aux_img/combined/{patient_id}_{idx}_combined.png"
            self._slice_cache[idx] = QtGui.QPixmap(path)
            # Keep cache under 50 entries
            if len(self._slice_cache) > 50:
                oldest = next(iter(self._slice_cache))
                del self._slice_cache[oldest]
        self.imageHeart.setPixmap(self._slice_cache[idx])

    def nextSlice(self):
        current = self.getCurrentSlice()
        new = min(current + 1, no_slices - 1)
        self.actual_slice.blockSignals(True)
        self.actual_slice.setValue(new)
        self.actual_slice.blockSignals(False)
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setValue(new)
        self.sliceSlider.blockSignals(False)
        self.displaySlice()

    def previousSlice(self):
        current = self.getCurrentSlice()
        new = max(current - 1, 1)
        self.actual_slice.blockSignals(True)
        self.actual_slice.setValue(new)
        self.actual_slice.blockSignals(False)
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setValue(new)
        self.sliceSlider.blockSignals(False)
        self.displaySlice()

    def sliderChanged(self, value):
        self.actual_slice.blockSignals(True)
        self.actual_slice.setValue(value)
        self.actual_slice.blockSignals(False)
        self.displaySlice()

    def spinChanged(self, value):
        self.sliceSlider.blockSignals(True)
        self.sliceSlider.setValue(value)
        self.sliceSlider.blockSignals(False)
        self.displaySlice()

    def editionHandler(self):
        current = self.getCurrentSlice()
        current_text = self.slices.toPlainText()
        if not current_text:
            self.slices.setText(str(current))
        else:
            self.slices.setText(f'{current_text}, {current}')

    # ── Retranslate / wire-up ─────────────────────────────────────────────────

    def retranslateUi(self, MainWindow):
        _t = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_t("MainWindow",
            "HARTA — Epicardial Fat Segmentation"))
        MainWindow.setWindowIcon(QtGui.QIcon(':/aux/logo_w.png'))
        MainWindow.setIconSize(QtCore.QSize(500, 200))

        self.openFile.setText(_t("MainWindow", "Open Exam…"))
        self.generateVolumeSemi.setText(_t("MainWindow", "Calculate New\nVolume"))
        self.slices_edited.setText(_t("MainWindow", "Slices edited:"))
        self.textAuto_2.setText(_t("MainWindow", "This may take a few seconds."))
        self.new_volume.setText(_t("MainWindow", "Epicardial Fat Volume (ml)"))
        self.view3DSemi.setText(_t("MainWindow", "3D View"))
        self.dicom_file.setText(_t("MainWindow", "DICOM Folder"))   # fix: was "DICOM File"
        self.generateVolumeAuto.setText(_t("MainWindow", "Calculate\nVolume"))
        self.view3DAuto.setText(_t("MainWindow", "3D View"))
        self.volume_detected.setText(_t("MainWindow", "Epicardial Fat Volume (ml)"))
        self.label_2.setText(_t("MainWindow", "Slice"))
        self.of.setText(_t("MainWindow", "of"))
        self.previous.setText(_t("MainWindow", "« Previous"))
        self.next.setText(_t("MainWindow", "Next »"))
        self.edit_slice.setText(_t("MainWindow", "✏  Edit Slice"))
        self.closeButton.setText(_t("MainWindow", "Close"))
        self.autoLabel.setText(_t("MainWindow", "Automatic Detection"))
        self.semiLabel.setText(_t("MainWindow", "Manual Adjustment"))  # fix: was "intervention"
        self.textAuto.setText(_t("MainWindow", "This may take a few seconds."))
        self.actionOpen_File.setText(_t("MainWindow", "Open File…"))

        # Hide initially
        self.generateVolumeAuto.hide()
        self.automatic_intervention.hide()
        self.manual_intervention.hide()
        self.infoAuto.hide()
        self.autoLabel.hide()
        self.semiLabel.hide()
        self.view3DAuto.hide()
        self.view3DSemi.hide()

        # Connect signals
        self.openFile.clicked.connect(self.searchFolder)
        self.generateVolumeAuto.clicked.connect(self.generateVolume)
        self.generateVolumeSemi.clicked.connect(self.generateNewVolume)
        self.edit_slice.clicked.connect(self.openWindow)
        self.next.clicked.connect(self.nextSlice)
        self.previous.clicked.connect(self.previousSlice)
        self.sliceSlider.valueChanged.connect(self.sliderChanged)
        self.actual_slice.valueChanged.connect(self.spinChanged)
        self.closeButton.clicked.connect(self.closeProgram)

        # Scroll-to-edit + double-click on CT image
        self.imageHeart.setNavigator(self)


# ══════════════════════════════════════════════════════════════════════════════
# SECOND WINDOW — Contour editor
# ══════════════════════════════════════════════════════════════════════════════
class SecondWindow(QWidget):
    submited = QtCore.pyqtSignal(int)

    def __init__(self, slice_edit, patient_edit):
        super().__init__()
        self.slice_edit   = slice_edit
        self.patient_edit = patient_edit
        self.eventhandler = EventHandler(self)
        self._orig = cv.imread(
            f"aux_img/slices/{patient_edit}_{slice_edit - 1}.png",
            cv.IMREAD_GRAYSCALE)
        if self._orig is None:
            # fallback: try with slice_edit (not -1) for backward compat
            self._orig = cv.imread(
                f"aux_img/slices/{patient_edit}_{slice_edit}.png",
                cv.IMREAD_GRAYSCALE)
        self._wl_center = 128
        self._wl_width  = 256
        self._has_unsaved_points = False
        self.setupUi()
        # Load auto-detected contour as ghost reference layer
        self.eventhandler.loadReferenceContour(patient_edit, slice_edit - 1)

    # ── Windowing ────────────────────────────────────────────────────────────

    def _applyWindowing(self):
        if self._orig is None:
            return
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

    # ── Mode selector ─────────────────────────────────────────────────────────

    def _setMode(self, mode):
        self.eventhandler.setMode(mode)
        active = ("QPushButton{border-radius:8px;background-color:rgb(70,136,244);"
                  "color:white;border:none;}QPushButton:hover{background-color:rgb(18,77,150);}")
        inactive = ("QPushButton{border-radius:8px;background-color:rgb(234,234,234);"
                    "color:black;border:none;}QPushButton:hover{background-color:rgb(183,181,181);}")
        for btn in (self._btnModeDraw, self._btnModeBrush,
                    self._btnModeEdit, self._btnModeWindow):
            btn.setStyleSheet(inactive)
        self._dragHint.hide()
        if mode == 'draw':
            self._btnModeDraw.setStyleSheet(active)
        elif mode == 'brush':
            self._btnModeBrush.setStyleSheet(active)
        elif mode == 'edit':
            self._btnModeEdit.setStyleSheet(active)
            self._editHint.show()
        elif mode == 'window':
            self._btnModeWindow.setStyleSheet(active)
            self._dragHint.show()
        if mode != 'edit':
            self._editHint.hide()

    # ── Propagation ──────────────────────────────────────────────────────────

    def _propagateContour(self):
        n, ok = QInputDialog.getInt(
            self, "Propagate Contour",
            "Copy contour to ±N adjacent slices\n(morphological adaptation per step):",
            value=2, min=1, max=10)
        if not ok:
            return
        src_path = f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png'
        if not os.path.exists(src_path):
            QMessageBox.warning(self, "No contour saved",
                                "Click Done first to save the contour, then Propagate.")
            return
        src_bgr  = cv.imread(src_path)
        src_gray = cv.cvtColor(src_bgr, cv.COLOR_BGR2GRAY)
        _, src_bin = cv.threshold(src_gray, 127, 255, cv.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        count = 0
        for offset in range(-n, n + 1):
            if offset == 0:
                continue
            target = int(self.slice_edit) - 1 + offset
            if target < 0 or target >= no_slices - 1:
                continue
            iters   = abs(offset)
            morphed = (cv.erode(src_bin, kernel, iterations=iters)
                       if offset > 0
                       else cv.dilate(src_bin, kernel, iterations=iters))
            morphed_bgr  = cv.cvtColor(morphed, cv.COLOR_GRAY2BGR)
            cv.imwrite(f'aux_img/contours/{self.patient_edit}_{target}_c.png', morphed_bgr)
            slice_path = f'aux_img/slices/{self.patient_edit}_{target}.png'
            if os.path.exists(slice_path):
                overlay = morphed_bgr.copy()
                overlay[morphed > 0] = [22, 9, 224]
                img_s = cv.imread(slice_path)
                if img_s is not None:
                    out = cv.addWeighted(overlay, 0.2, img_s, 1, 0, img_s)
                    cv.imwrite(
                        f'aux_img/combined/{self.patient_edit}_{target}_combined.png', out)
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
        elif key == Qt.Key_V:
            self._setMode('edit')
        elif key == Qt.Key_W:
            self._setMode('window')
        elif key == Qt.Key_R:
            self.eventhandler.toggleReference()
        elif key == Qt.Key_Z and mods & Qt.ControlModifier:
            self.eventhandler.undoLastPoint()
        elif key == Qt.Key_Y and mods & Qt.ControlModifier:
            self.eventhandler.redo()
        elif key == Qt.Key_Z and mods & (Qt.ControlModifier | Qt.ShiftModifier):
            self.eventhandler.redo()
        elif key == Qt.Key_Delete:
            self.eventhandler.deleteHoveredPoint()
        elif key == Qt.Key_1:
            self._applyPreset(128, 256)
        elif key == Qt.Key_2:
            self._applyPreset(128, 150)
        elif key == Qt.Key_3:
            self._applyPreset(210,  80)
        elif key == Qt.Key_4:
            self._applyPreset( 64, 180)
        else:
            super().keyPressEvent(event)

    # ── Confirm on close ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.eventhandler.points and not self._has_unsaved_points:
            reply = QMessageBox.question(
                self, "Discard changes?",
                "You have an unsaved contour. Close without saving?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel)
            if reply == QMessageBox.Save:
                self.onSubmit()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # ── UI setup ─────────────────────────────────────────────────────────────

    def setupUi(self):
        self.setWindowTitle(f"Contour Editor — Slice {self.slice_edit}")
        self.setStyleSheet("background-color:rgb(255,255,255);")
        self.setWindowIcon(QtGui.QIcon(':/aux/logo_w.png'))
        self.resize(1050, 750)
        self.setMinimumSize(QtCore.QSize(1050, 750))

        buttonStyle = ("QPushButton{border-style:solid;border-width:0px;border-radius:10px;"
                       "background-color:rgb(234,234,234);}"
                       "QPushButton::hover{background-color:rgb(183,181,181);}")
        presetStyle = ("QPushButton{border-style:solid;border-width:1px;"
                       "border-color:rgb(200,200,200);border-radius:8px;"
                       "background-color:rgb(245,245,245);}"
                       "QPushButton::hover{background-color:rgb(70,136,244);"
                       "color:white;border-color:rgb(70,136,244);}")

        fnt9 = QtGui.QFont(); fnt9.setFamily("Roboto"); fnt9.setPointSize(9)
        fntB = QtGui.QFont(); fntB.setFamily("Roboto"); fntB.setPointSize(10); fntB.setBold(True)
        fntS = QtGui.QFont(); fntS.setFamily("Roboto"); fntS.setPointSize(8)

        PX = 560   # right-panel x

        # ── Image canvas ─────────────────────────────────────────────────────
        self.img = self.eventhandler
        self._applyWindowing()
        self.img.setGeometry(QRect(20, 65, 512, 512))
        self.img.setMinimumSize(QtCore.QSize(512, 512))
        self.img.setScaledContents(True)
        self.img.setCursor(Qt.CrossCursor)

        # ── Mode row (4 modes) ───────────────────────────────────────────────
        modeActive = ("QPushButton{border-radius:8px;background-color:rgb(70,136,244);"
                      "color:white;border:none;}"
                      "QPushButton:hover{background-color:rgb(18,77,150);}")
        modeInactive = ("QPushButton{border-radius:8px;background-color:rgb(234,234,234);"
                        "color:black;border:none;}"
                        "QPushButton:hover{background-color:rgb(183,181,181);}")

        QLabel("Mode:", self).setGeometry(QRect(PX, 70, 50, 26))

        self._btnModeDraw = QPushButton("✏  Draw", self)
        self._btnModeDraw.setGeometry(QRect(PX + 55, 65, 88, 30))
        self._btnModeDraw.setFont(fnt9)
        self._btnModeDraw.setStyleSheet(modeActive)
        self._btnModeDraw.setToolTip("Click để đặt điểm contour  (D)")
        self._btnModeDraw.clicked.connect(lambda: self._setMode('draw'))

        self._btnModeBrush = QPushButton("🖌  Brush", self)
        self._btnModeBrush.setGeometry(QRect(PX + 150, 65, 88, 30))
        self._btnModeBrush.setFont(fnt9)
        self._btnModeBrush.setStyleSheet(modeInactive)
        self._btnModeBrush.setToolTip("Kéo vẽ tự do — tự snap vào cạnh  (B)")
        self._btnModeBrush.clicked.connect(lambda: self._setMode('brush'))

        self._btnModeEdit = QPushButton("↖  Edit Pts", self)
        self._btnModeEdit.setGeometry(QRect(PX + 245, 65, 100, 30))
        self._btnModeEdit.setFont(fnt9)
        self._btnModeEdit.setStyleSheet(modeInactive)
        self._btnModeEdit.setToolTip(
            "Kéo điểm • Right-click xóa • Click segment chèn  (V)")
        self._btnModeEdit.clicked.connect(lambda: self._setMode('edit'))

        self._btnModeWindow = QPushButton("🖱  Window", self)
        self._btnModeWindow.setGeometry(QRect(PX + 352, 65, 100, 30))
        self._btnModeWindow.setFont(fnt9)
        self._btnModeWindow.setStyleSheet(modeInactive)
        self._btnModeWindow.setToolTip("Kéo: ←→ Width  ↑↓ Center  (W)")
        self._btnModeWindow.clicked.connect(lambda: self._setMode('window'))

        self._dragHint = QLabel("Drag: ←→ Width   ↑↓ Center", self)
        self._dragHint.setGeometry(QRect(PX + 55, 99, 300, 16))
        self._dragHint.setFont(fntS)
        self._dragHint.setStyleSheet("color:rgb(100,140,220);")
        self._dragHint.hide()

        self._editHint = QLabel(
            "↖ Drag point  •  Right-click delete  •  Click segment to insert", self)
        self._editHint.setGeometry(QRect(PX + 55, 99, 400, 16))
        self._editHint.setFont(fntS)
        self._editHint.setStyleSheet("color:rgb(70,160,70);")
        self._editHint.hide()

        sep0 = QLabel(self)
        sep0.setGeometry(QRect(PX, 122, 460, 2))
        sep0.setStyleSheet("background:rgb(220,220,220);")

        # ── Windowing panel ───────────────────────────────────────────────────
        wTitle = QLabel("Windowing / Gray-level Mapping", self)
        wTitle.setGeometry(QRect(PX, 132, 460, 26)); wTitle.setFont(fntB)

        sep1 = QLabel(self); sep1.setGeometry(QRect(PX, 161, 460, 2))
        sep1.setStyleSheet("background:rgb(220,220,220);")

        QLabel("Window Center (Level)", self).setGeometry(QRect(PX, 171, 220, 18))
        self._centerSlider = QSlider(Qt.Horizontal, self)
        self._centerSlider.setGeometry(QRect(PX, 193, 340, 22))
        self._centerSlider.setRange(0, 255); self._centerSlider.setValue(128)
        self._centerSpin = QSpinBox(self)
        self._centerSpin.setGeometry(QRect(PX + 350, 189, 72, 30))
        self._centerSpin.setRange(0, 255); self._centerSpin.setValue(128)
        self._centerSpin.setFont(fnt9)

        QLabel("Window Width", self).setGeometry(QRect(PX, 233, 220, 18))
        self._widthSlider = QSlider(Qt.Horizontal, self)
        self._widthSlider.setGeometry(QRect(PX, 255, 340, 22))
        self._widthSlider.setRange(1, 256); self._widthSlider.setValue(256)
        self._widthSpin = QSpinBox(self)
        self._widthSpin.setGeometry(QRect(PX + 350, 251, 72, 30))
        self._widthSpin.setRange(1, 256); self._widthSpin.setValue(256)
        self._widthSpin.setFont(fnt9)

        QLabel("Presets:", self).setGeometry(QRect(PX, 295, 70, 18))
        for i, (lbl, c, w) in enumerate([
                ("Full view", 128, 256), ("Soft Tissue", 128, 150),
                ("Bone", 210, 80),       ("Lung", 64, 180)]):
            btn = QPushButton(lbl, self)
            btn.setGeometry(QRect(PX + i * 110, 317, 100, 28))
            btn.setFont(fnt9); btn.setStyleSheet(presetStyle)
            btn.clicked.connect(lambda _, c=c, w=w: self._applyPreset(c, w))

        sep2 = QLabel(self); sep2.setGeometry(QRect(PX, 358, 460, 2))
        sep2.setStyleSheet("background:rgb(220,220,220);")

        # ── Instructions ──────────────────────────────────────────────────────
        instrLbl = QLabel(
            "✏ Draw: click to place points\n"
            "🖌 Brush: drag freehand → auto-snaps to edges\n"
            "↖ Edit Pts: drag/right-click/segment-click to adjust\n"
            "🖱 Window: drag on image to adjust contrast", self)
        instrLbl.setGeometry(QRect(PX, 368, 440, 68))
        instrLbl.setFont(fntS)
        instrLbl.setStyleSheet("color:rgb(120,120,120);")

        # ── Contour tools ──────────────────────────────────────────────────────
        sep3 = QLabel(self); sep3.setGeometry(QRect(PX, 446, 460, 2))
        sep3.setStyleSheet("background:rgb(220,220,220);")

        ctTitle = QLabel("Contour Tools", self)
        ctTitle.setGeometry(QRect(PX, 456, 460, 22)); ctTitle.setFont(fntB)

        propBtn = QPushButton("↗  Propagate ±N slices", self)
        propBtn.setGeometry(QRect(PX, 484, 200, 30))
        propBtn.setFont(fnt9); propBtn.setStyleSheet(buttonStyle)
        propBtn.setToolTip("Copy contour to adjacent slices with morphological adaptation")
        propBtn.clicked.connect(self._propagateContour)

        refBtn = QPushButton("👁  Toggle Reference", self)
        refBtn.setGeometry(QRect(PX + 215, 484, 160, 30))
        refBtn.setFont(fnt9); refBtn.setStyleSheet(buttonStyle)
        refBtn.setToolTip("Show/hide auto-detected contour as ghost overlay  (R)")
        refBtn.clicked.connect(self.eventhandler.toggleReference)

        # ── Hotkeys reference ──────────────────────────────────────────────────
        sep4 = QLabel(self); sep4.setGeometry(QRect(PX, 526, 460, 2))
        sep4.setStyleSheet("background:rgb(220,220,220);")

        hkTitle = QLabel("Keyboard Shortcuts", self)
        hkTitle.setGeometry(QRect(PX, 536, 460, 22)); hkTitle.setFont(fntB)

        hkLbl = QLabel(
            "D Draw  •  B Brush  •  V Edit Points  •  W Window\n"
            "R Toggle reference  •  Del Delete hovered point\n"
            "Ctrl+Z Undo  •  Ctrl+Y / Ctrl+Shift+Z Redo\n"
            "1-4 Presets  •  Enter Done  •  Esc Cancel", self)
        hkLbl.setGeometry(QRect(PX, 562, 460, 64))
        hkLbl.setFont(fntS)
        hkLbl.setStyleSheet("color:rgb(80,80,80);")

        # Connect windowing signals
        self._centerSlider.valueChanged.connect(self._onCenterChanged)
        self._centerSpin.valueChanged.connect(self._onCenterSpinChanged)
        self._widthSlider.valueChanged.connect(self._onWidthChanged)
        self._widthSpin.valueChanged.connect(self._onWidthSpinChanged)

        # ── Done / Cancel ─────────────────────────────────────────────────────
        self.doneButton = QPushButton('Done', self)
        self.doneButton.setGeometry(QRect(820, 700, 100, 32))
        self.doneButton.setFont(fnt9); self.doneButton.setStyleSheet(buttonStyle)
        self.doneButton.clicked.connect(self.onSubmit)

        self.cancelButton = QPushButton('Cancel', self)
        self.cancelButton.setGeometry(QRect(930, 700, 100, 32))
        self.cancelButton.setFont(fnt9); self.cancelButton.setStyleSheet(buttonStyle)
        self.cancelButton.clicked.connect(self.close)

        self.show()

    # ── Submit ────────────────────────────────────────────────────────────────

    def onSubmit(self):
        path = self.eventhandler.getPath()
        self.image = QImage(800, 800, QImage.Format_RGB32)
        self.image.fill(Qt.black)
        painter = QPainter(self.image)
        painter.drawImage(self.rect(), self.image, self.image.rect())
        painter.setPen(Qt.white)
        painter.fillPath(path, QBrush(QColor("white")))
        painter.drawPath(path)

        self._has_unsaved_points = True
        self.submited.emit(self.slice_edit)
        filePath = f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png'
        self.image.save(filePath)
        self.rescaleImage()
        self.combineImages()
        self.eventhandler.clearPath()
        self.close()

    def rescaleImage(self):
        fp = f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png'
        img = cv.imread(fp)
        cv.imwrite(fp, cv.resize(img, (512, 512), interpolation=cv.INTER_NEAREST))

    def combineImages(self):
        mask = cv.imread(
            f'aux_img/contours/{self.patient_edit}_{int(self.slice_edit) - 1}_c.png')
        img  = cv.imread(
            f'aux_img/slices/{self.patient_edit}_{int(self.slice_edit) - 1}.png')
        mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [22, 9, 224]
        out = cv.addWeighted(mask, 0.2, img, 1, 0, img)
        cv.imwrite(
            f'aux_img/combined/{self.patient_edit}_{int(self.slice_edit) - 1}_combined.png',
            out)


# ══════════════════════════════════════════════════════════════════════════════
# EVENT HANDLER — interactive contour canvas
# ══════════════════════════════════════════════════════════════════════════════
class EventHandler(QLabel):
    """Four interaction modes:
       draw   – click to place contour points
       brush  – drag freehand; on release snaps to Canny edges (Magnetic Lasso)
       edit   – drag existing points, right-click delete, click-segment insert
       window – drag to adjust windowing (dx→Width, dy→Center)
    """

    HIT_RADIUS = 12   # px manhattan distance for point hit-test
    MAX_UNDO   = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flag   = True
        self.points = []
        self.path   = QtGui.QPainterPath()
        self._path_dirty = True

        self._mode        = 'draw'
        self._drag_start  = None
        self._drag_center = 128
        self._drag_width  = 256
        self._brushing    = False

        # Edit mode
        self._edit_drag_idx = -1
        self._hover_pos     = None   # last known mouse position (for Del key)

        # Undo / redo
        self._undo_stack = []   # list[list[QPoint]]
        self._redo_stack = []

        # Reference (auto-detected) contour layer
        self._ref_points = []
        self._show_ref   = True

    # ── Mode ────────────────────────────────────────────────────────────────

    def setMode(self, mode):
        self._mode = mode
        cursors = {'draw': Qt.CrossCursor, 'brush': Qt.PointingHandCursor,
                   'edit': Qt.ArrowCursor,  'window': Qt.SizeAllCursor}
        self.setCursor(cursors.get(mode, Qt.CrossCursor))

    # ── Reference contour ────────────────────────────────────────────────────

    def loadReferenceContour(self, patient_id, slice_idx):
        """Load existing saved contour as an orange ghost overlay."""
        path = f'aux_img/contours/{patient_id}_{slice_idx}_c.png'
        if not os.path.exists(path):
            self._ref_points = []; return
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            self._ref_points = []; return
        _, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._ref_points = []; return
        largest = max(contours, key=cv.contourArea)
        h, w = img.shape
        ww, wh = max(self.width(), 512), max(self.height(), 512)
        self._ref_points = [
            QtCore.QPoint(int(p[0][0] * ww / w), int(p[0][1] * wh / h))
            for p in largest]
        self.update()

    def toggleReference(self):
        self._show_ref = not self._show_ref
        self.update()

    # ── Undo / Redo ──────────────────────────────────────────────────────────

    def _pushUndo(self):
        self._undo_stack.append([QtCore.QPoint(p) for p in self.points])
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undoLastPoint(self):
        if self._undo_stack:
            self._redo_stack.append([QtCore.QPoint(p) for p in self.points])
            self.points      = self._undo_stack.pop()
            self._path_dirty = True
            self.update()

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append([QtCore.QPoint(p) for p in self.points])
            self.points      = self._redo_stack.pop()
            self._path_dirty = True
            self.update()

    # ── Hit testing ──────────────────────────────────────────────────────────

    def _hitPoint(self, pos):
        """Return index of nearest point within HIT_RADIUS, or -1."""
        px, py = pos.x(), pos.y()
        for i, p in enumerate(self.points):
            if abs(px - p.x()) + abs(py - p.y()) <= self.HIT_RADIUS:
                return i
        return -1

    def _hitSegment(self, pos):
        """Return index j such that pos lies near segment j→j+1, or -1."""
        if len(self.points) < 2:
            return -1
        px, py     = pos.x(), pos.y()
        best_j, bd = -1, self.HIT_RADIUS * 2
        n = len(self.points)
        for j in range(n):
            a = self.points[j];  b = self.points[(j + 1) % n]
            dx, dy = b.x() - a.x(), b.y() - a.y()
            sq = dx * dx + dy * dy
            if sq == 0:
                continue
            t  = max(0.0, min(1.0, ((px - a.x()) * dx + (py - a.y()) * dy) / sq))
            cx = a.x() + t * dx;  cy = a.y() + t * dy
            d  = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < bd:
                bd, best_j = d, j
        return best_j

    def deleteHoveredPoint(self):
        """Delete the point nearest to last known mouse pos (for Del key)."""
        if self._hover_pos is None or len(self.points) <= 3:
            return
        hit = self._hitPoint(self._hover_pos)
        if hit < 0:
            # Fallback: find globally nearest
            px, py = self._hover_pos.x(), self._hover_pos.y()
            hit = min(range(len(self.points)),
                      key=lambda i: (self.points[i].x()-px)**2+(self.points[i].y()-py)**2)
        self._pushUndo()
        self.points.pop(hit)
        self._path_dirty = True
        self.update()

    # ── Mouse events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        pos = event.pos()
        self._hover_pos = pos

        if self._mode == 'draw':
            if event.button() == Qt.LeftButton:
                self._pushUndo()
                self.points.append(pos)
                self._path_dirty = True
                self.update()

        elif self._mode == 'edit':
            if event.button() == Qt.LeftButton:
                hit = self._hitPoint(pos)
                if hit >= 0:
                    self._pushUndo()
                    self._edit_drag_idx = hit
                else:
                    seg = self._hitSegment(pos)
                    if seg >= 0:
                        self._pushUndo()
                        self.points.insert(seg + 1, QtCore.QPoint(pos))
                        self._edit_drag_idx = seg + 1
                        self._path_dirty = True
                        self.update()
            elif event.button() == Qt.RightButton:
                hit = self._hitPoint(pos)
                if hit >= 0 and len(self.points) > 3:
                    self._pushUndo()
                    self.points.pop(hit)
                    self._path_dirty = True
                    self.update()

        elif self._mode == 'brush':
            self._brushing = True
            self.points    = []
            self.points.append(pos)
            self._path_dirty = True
            self.update()

        else:  # window
            self._drag_start = pos
            sw = self.parent()
            if hasattr(sw, '_wl_center'):
                self._drag_center = sw._wl_center
                self._drag_width  = sw._wl_width

    def mouseMoveEvent(self, event):
        pos = event.pos()
        self._hover_pos = pos

        if self._mode == 'edit' and self._edit_drag_idx >= 0:
            self.points[self._edit_drag_idx] = pos
            self._path_dirty = True
            self.update()

        elif self._mode == 'brush' and self._brushing:
            self.points.append(pos)
            self._path_dirty = True
            self.update()

        elif self._mode == 'window' and self._drag_start is not None:
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            sw = self.parent()
            if hasattr(sw, '_applyPreset'):
                sw._applyPreset(
                    max(0,   min(255, int(self._drag_center - dy))),
                    max(1,   min(256, int(self._drag_width  + dx))))

    def mouseReleaseEvent(self, event):
        if self._mode == 'edit':
            self._edit_drag_idx = -1
        elif self._mode == 'brush' and self._brushing:
            self._brushing   = False
            self._simplifyPoints()
            self._snapToEdges()
            self._path_dirty = True
            self.update()
        self._drag_start = None

    # ── Smart Brush helpers ──────────────────────────────────────────────────

    def _simplifyPoints(self, epsilon=3.0):
        if len(self.points) < 3:
            return
        pts = np.array([[p.x(), p.y()] for p in self.points],
                       dtype=np.float32).reshape(-1, 1, 2)
        simp = cv.approxPolyDP(pts, epsilon, closed=True)
        self.points = [QtCore.QPoint(int(p[0][0]), int(p[0][1])) for p in simp]

    def _snapToEdges(self, radius=20):
        parent = self.parent()
        if not hasattr(parent, '_orig') or parent._orig is None:
            return
        orig   = parent._orig
        img_h, img_w = orig.shape
        ww, wh = self.width(), self.height()
        edges  = cv.Canny(orig, 30, 100)
        edges  = cv.dilate(edges, np.ones((5, 5), np.uint8))
        snapped = []
        for p in self.points:
            ix = max(0, min(img_w - 1, int(p.x() * img_w / ww)))
            iy = max(0, min(img_h - 1, int(p.y() * img_h / wh)))
            x1, x2 = max(0, ix - radius), min(img_w, ix + radius + 1)
            y1, y2 = max(0, iy - radius), min(img_h, iy + radius + 1)
            ey = np.argwhere(edges[y1:y2, x1:x2] > 0)
            if len(ey) > 0:
                d    = np.hypot(ey[:, 1] - (ix - x1), ey[:, 0] - (iy - y1))
                best = ey[np.argmin(d)]
                snapped.append(QtCore.QPoint(
                    int((x1 + best[1]) * ww / img_w),
                    int((y1 + best[0]) * wh / img_h)))
            else:
                snapped.append(p)
        self.points = snapped

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.flag:
            self.points      = []
            self.path        = QtGui.QPainterPath()
            self._path_dirty = True
            self.flag        = False

        painter = QPainter(self)

        # Ghost reference contour (orange dashed)
        if self._show_ref and len(self._ref_points) > 1:
            painter.setPen(QPen(QColor(255, 140, 0, 140), 2, Qt.DashLine))
            n = len(self._ref_points)
            for i in range(n):
                painter.drawLine(self._ref_points[i],
                                 self._ref_points[(i + 1) % n])

        # Current contour
        if self._mode == 'edit':
            for i, pos in enumerate(self.points):
                if i == self._edit_drag_idx:
                    painter.setPen(QPen(QColor(255, 200, 0), 2))
                    painter.setBrush(QColor(255, 200, 0))
                else:
                    painter.setPen(QPen(QColor(224, 9, 22), 2))
                    painter.setBrush(QColor(224, 9, 22, 200))
                painter.drawEllipse(pos, 5, 5)
            painter.setBrush(Qt.NoBrush)
        else:
            painter.setPen(QPen(QColor(224, 9, 22), 10, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            for pos in self.points:
                painter.drawPoint(pos)

        if len(self.points) > 2:
            if self._path_dirty:
                self.buildPath()
                self._path_dirty = False
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(70, 136, 244), 3, Qt.SolidLine))
            painter.drawPath(self.path)

    # ── Path ────────────────────────────────────────────────────────────────

    def buildPath(self):
        factor = 0.5
        self.path = QtGui.QPainterPath(self.points[0])
        cp1 = None
        for p, current in enumerate(self.points[1:-1], 1):
            source      = QtCore.QLineF(self.points[p - 1], current)
            target      = QtCore.QLineF(current, self.points[p + 1])
            targetAngle = target.angleTo(source)
            if targetAngle > 180:
                angle = (source.angle() + source.angleTo(target) / 2) % 360
            else:
                angle = (target.angle() + target.angleTo(source) / 2) % 360
            revTarget = QtCore.QLineF.fromPolar(
                source.length() * factor, angle + 180).translated(current)
            cp2 = revTarget.p2()
            if p == 1:
                self.path.quadTo(cp2, current)
            else:
                self.path.cubicTo(cp1, cp2, current)
            revSource = QtCore.QLineF.fromPolar(
                target.length() * factor, angle).translated(current)
            cp1 = revSource.p2()
        if cp1 is not None:
            self.path.quadTo(cp1, self.points[-1])

    def getPath(self):
        return self.path

    def clearPath(self):
        self.flag        = True
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._edit_drag_idx = -1


# ══════════════════════════════════════════════════════════════════════════════
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
    for d in ['aux_img/combined', 'aux_img/slices', 'aux_img/contours', 'aux_img/fat']:
        os.makedirs(d, exist_ok=True)

    sys.excepthook = _safe_excepthook

    app = QtWidgets.QApplication(sys.argv)

    import traceback
    def _qt_hook(exc_type, exc_value, exc_tb):
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.stderr)
        box = QMessageBox()
        box.setWindowTitle("Error")
        box.setText(f"{exc_type.__name__}: {exc_value}")
        box.setDetailedText(msg)
        box.exec_()
    sys.excepthook = _qt_hook

    MainWindow = MainWindowClass()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setUi(ui)
    MainWindow.show()
    sys.exit(app.exec_())
