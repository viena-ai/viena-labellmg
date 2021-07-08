#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import distutils.spawn
import os.path
import platform
import re
import shutil
import sys
import subprocess
import webbrowser
from pathlib import Path
from functools import partial
from collections import defaultdict
from typing import List

import cv2
import numpy as np
import argparse
import shutil,os,zipfile
from pathlib import Path

import boto3
import requests
import json
import uuid
import glob
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import (QPixmap)
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.ustr import ustr
from libs.createtrainingdata import *
from libs.startTrainAutomatically import *

from libs.hashableQListWidgetItem import HashableQListWidgetItem
from tqdm import tqdm
import urllib.request

__appname__ = 'labelImg'
global checkfortrain
global weightName
global weightUrl
global previousDownloadedWeightFile
weightName = []
weightUrl = []
checkfortrain=0
previousDownloadedWeightFile=''
class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)
        print("*****getStr****")
        print(getStr)

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        self.usingPascalVocFormat = True
        self.usingYoloFormat = False

        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(getStr('useDefaultLabel'))
        self.useDefaultLabelCheckbox.setChecked(True)
        self.defaultLabelTextLine = QLineEdit()
        defaultClass = self.loadDefaultClassName(defaultPrefdefClassFile)
        defaultClass = defaultClass.strip()
        print(defaultClass)
        self.defaultLabelTextLine.setText(defaultClass)

        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)


        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(getStr('useDifficult'))
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Create and add combobox for showing unique labels in group
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(self.comboBox)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.tabscroll = QScrollArea()
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.tabsnav = QTabWidget()
        self.tabnav1 = QWidget()
        self.tabnav2 = QWidget()
        self.tabnav3 = QWidget()
        self.tabnav4 = QWidget()



        self.tabsnav.addTab(self.tabnav1, "Training Details")
        self.tabsnav.addTab(self.tabnav2, "Weights")
        self.tabsnav.addTab(self.tabnav3, "Logs")
        self.tabsnav.addTab(self.tabnav4, "Auto annotate")


        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))
        self.tabs.addTab(self.tab1, "Images")
        self.tabs.addTab(self.tab2, "Annotation")
        self.tabs.addTab(self.canvas, "Tab 2")
        self.tabs.addTab(self.tabsnav, "Training")

        # Create first tab
        self.tab1.layout = QVBoxLayout(self)
        OpenDir = self.lastOpenDir


        self.createGridLayout(OpenDir)

        self.tab2.layout = QVBoxLayout(self)
        self.tab2.setLayout(self.tab2.layout)

        self.createTrainingDetailsForm()
        self.tabnav1.layout = QVBoxLayout(self)
        self.tabnav1.layout.addWidget(self.formGroupBox)
        self.tabnav1.setLayout(self.tabnav1.layout)

        self.weightDetailsForm()
        self.logDetailsbox = QVBoxLayout()
        self.logsRefreshButton = QPushButton("Refresh")
        self.logsRefreshButton.setFixedHeight(25)
        self.logsRefreshButton.setFixedWidth(100)
        self.logsRefreshButton.clicked.connect(lambda checked: self.getdatafromEndpoint())
        self.logDetailsbox.addWidget(self.logsRefreshButton)


        self.logOutput = QTextEdit()
        self.logOutput.setReadOnly(True)
        self.logsDetailsForm()


        self.autoannotateDetailsbox = QVBoxLayout()
        self.autoAnnotateOutput = QTextEdit()
        self.autoAnnotateOutput.setReadOnly(True)
        self.AutouploadButton = QPushButton("Auto train")
        self.AutouploadButton.setFixedHeight(25)
        self.AutouploadButton.setFixedWidth(100)
        self.AutouploadButton.clicked.connect(lambda checked: self.autoUploadandTrain())
        self.autoannotateDetailsbox.addWidget(self.AutouploadButton)
        self.AutoAnnotateButton = QPushButton("Auto Annotate")
        self.AutoAnnotateButton.setFixedHeight(25)
        self.AutoAnnotateButton.setFixedWidth(100)
        self.AutoAnnotateButton.clicked.connect(lambda checked: self.getAutoTrainWeights())
        self.autoannotateDetailsbox.addWidget(self.AutoAnnotateButton)
        self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
        self.tabnav4.setLayout(self.autoannotateDetailsbox)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }

        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)



        # Add tabs to widget

        self.canvas.layout = QVBoxLayout(self)
        self.tab2.layout.addWidget(scroll)
        self.tab2.setLayout(self.tab2.layout)

        self.setCentralWidget(self.tabs)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)


        # Actions
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+a', 'open', getStr('openFileDetail'))

        annotate = action(getStr('startTraining'), self.annotate,
                      'Ctrl+a', 'startTraining', getStr('startTraining'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        changeSavedir = action(getStr('changeSaveDir'), self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action(getStr('nextImg'), self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))
        print(getStr('nextImgDetail'))

        openPrevImg = action(getStr('prevImg'), self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action(getStr('verifyImg'), self.verifyImg,
                        'space', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        save_format = action('&PascalVOC', self.change_format,
                             'Ctrl+', 'format_voc', getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        deleteImg = action(getStr('deleteImg'), self.deleteImg, 'Ctrl+D', 'close', getStr('deleteImgDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=open, close=close,
                              resetAll=resetAll, deleteImg=deleteImg,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  annotate, open, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open1 &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        addActions(self.menus.file,
                   (annotate, open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles, save, save_format,
                    saveAs, close, resetAll, deleteImg, quit))
        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            annotate, open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, None, create,
            copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            annotate, open, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.lastOpenDir = None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

    def createTrainingDetailsForm(self):
        self.formGroupBox = QGroupBox("Form layout")
        self.trainingLayout = QFormLayout()
        self.projectPath = QLineEdit("")
        self.imagePath = QLineEdit("")
        self.labelPath = QLineEdit("")
        self.classPath = QLineEdit("")
        self.noOfClassesPath = QLineEdit("")

        self.projectUploadButton = QPushButton("")
        self.projectUploadButton.resize(200, 150)
        self.projectUploadButton.setIcon(QIcon('resources/icons/folder.jpg'))

        self.imageUploadButton = QPushButton("")
        self.imageUploadButton.resize(200, 150)
        self.imageUploadButton.setIcon(QIcon('resources/icons/folder.jpg'))

        self.labelUploadButton = QPushButton("")
        self.labelUploadButton.resize(200, 150)
        self.labelUploadButton.setIcon(QIcon('resources/icons/folder.jpg'))

        self.classUploadButton = QPushButton("")
        self.classUploadButton.resize(200, 150)
        self.classUploadButton.setIcon(QIcon('resources/icons/folder.jpg'))


        self.formLayout = QGridLayout()
        self.formLayout.addWidget(QLabel("Project Directory:"), 0, 0)
        self.formLayout.addWidget(self.projectPath, 0, 1)
        self.formLayout.addWidget(self.projectUploadButton, 0, 2)

        self.formLayout.addWidget(QLabel("Images:"), 1, 0)
        self.formLayout.addWidget(self.imagePath, 1, 1)
        self.formLayout.addWidget(self.imageUploadButton, 1, 2)

        self.formLayout.addWidget(QLabel("Labels:"), 2, 0)
        self.formLayout.addWidget(self.labelPath, 2, 1)
        self.formLayout.addWidget(self.labelUploadButton, 2, 2)

        self.formLayout.addWidget(QLabel("Class Names (.txt):"), 3, 0)
        self.formLayout.addWidget(self.classPath, 3, 1)
        self.formLayout.addWidget(self.classUploadButton, 3, 2)

        self.formLayout.addWidget(QLabel("No of classes:"), 4, 0)
        self.formLayout.addWidget(self.noOfClassesPath, 4, 1)

        self.saveButton = QPushButton("Save")
        self.startTrainingButton = QPushButton("Start Training")
        self.weightsButton = QPushButton("Weights")


        self.trainingLayout.addRow(self.formLayout)

        self.buttonlayout = QGridLayout()
        self.buttonlayout.addWidget(self.saveButton, 0, 6)
        self.buttonlayout.addWidget(self.startTrainingButton, 0, 7)
        self.buttonlayout.addWidget(self.weightsButton, 0, 8)
        self.trainingLayout.addRow(self.buttonlayout)


        self.formGroupBox.setLayout(self.trainingLayout)

        self.projectUploadButton.clicked.connect(lambda checked: self.openProjectDirDialog())
        self.imageUploadButton.clicked.connect(lambda checked: self.openImagesDirDialog())
        self.labelUploadButton.clicked.connect(lambda checked: self.openLabelsDirDialog())
        self.classUploadButton.clicked.connect(lambda checked: self.openClassesDirDialog())
        self.startTrainingButton.clicked.connect(lambda checked: self.getDetailsForTraining())
    def getdatafromEndpoint(self):
        resp =''
        statusCode=100
        try:
            resp = requests.get(
                'http://192.168.1.5:8080/vflow/download_high_priority_req_res_msgs?device_code=ABCD_1')
            statusCode =resp.status_code
        except:
            print("No response from endpoint")
        data=[{}]
        print("data")
        if (statusCode == 200):
            try:
                data = resp.json()
                print(data)
            except:
                print("Empty json")
        try:
            for items in data:
                for item in items["weights"]:
                    if (item["name"] != ''):
                        weightName.append(item['name'])
                        weightUrl.append(item['url'])
        except:
            print("Empty weights")

        if (len(weightName) != 0):
            f = open('previous_conf.txt', 'w')
            for i in range(len(weightName)):
                f.write(weightName[i]+","+weightUrl[i]+'\n')  # python will convert \n to os.linesep
            f.close()  # you can omit in most cases as the destructor will call it

            self.table.setColumnCount(2)
            self.table.setRowCount(len(weightName))

            for index in range(len(weightName)):
                item1 = QTableWidgetItem(weightName[index])
                self.table.setItem(index, 0, item1)
                self.btn_sell = QPushButton('Download')
                self.btn_sell.clicked.connect(lambda checked, file=weightUrl[index]: self.downloadSelectedWeight(file))
                self.table.setCellWidget(index, 1, self.btn_sell)

        self.weightsbox.addWidget(self.table)
        self.tabnav2.setLayout(self.weightsbox)
        logs=''
        logsdata=[{}]
        for element in data:
            element.pop('weights', None)
            logsdata=data
        try:
            logslist = {k: v for d in logsdata for k, v in d.items()}
            logs = logslist['logs']
        except:
            print("Empty logs")

        print(logs)

        if(logs!= ''):
            font = self.logOutput.font()
            font.setFamily("Courier")
            font.setPointSize(15)
            self.logOutput.append(logs)

            self.logDetailsbox.addWidget(self.logOutput)
            self.tabnav3.setLayout(self.logDetailsbox)

    def weightDetailsForm(self):
        self.weightsbox = QVBoxLayout()
        self.weightsRefreshButton = QPushButton("Refresh")
        self.weightsRefreshButton.setFixedHeight(25)
        self.weightsRefreshButton.setFixedWidth(100)
        self.weightsRefreshButton.clicked.connect(lambda checked: self.getdatafromEndpoint())
        self.weightsbox.addWidget(self.weightsRefreshButton)
        # add table
        self.table = QTableWidget(self)

        data = [{ }]
        if(len(data) == 0):
            for items in data:
                for item in items["weights"]:
                    if (item["name"] != ''):
                        weightName.append(item['name'])
                        weightUrl.append(item['url'])

        if(len(weightName) !=0):
            self.table.setColumnCount(2)
            self.table.setRowCount(len(weightName))

            for index in range(len(weightName)):
                item1 = QTableWidgetItem(weightName[index])
                self.table.setItem(index, 0, item1)
                self.btn_sell = QPushButton('Download')
                self.btn_sell.clicked.connect(lambda checked,file=weightUrl[index]: self.downloadSelectedWeight(file))
                self.table.setCellWidget(index, 1, self.btn_sell)

        self.weightsbox.addWidget(self.table)
        self.tabnav2.setLayout(self.weightsbox)

    def logsDetailsForm(self):
        logs=''
        font = self.logOutput.font()
        font.setFamily("Courier")
        font.setPointSize(15)
        self.logOutput.append(logs)

        self.logDetailsbox.addWidget(self.logOutput)
        self.tabnav3.setLayout(self.logDetailsbox)

    def logsRefresh(self):
        self.logsDetailsForm()

    def weightsRefresh(self):
        self.weightDetailsForm()

    def downloadSelectedWeight(self, link):
        url = QUrl(link)
        QDesktopServices.openUrl(url)
        print(link)

    def openProjectDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return
        defaultOpenDir = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            projectDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDir,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        self.projectPath.setText(projectDirPath)

    def openImagesDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return
        defaultOpenDir = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            imageDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                 '%s - Open Directory' % __appname__,
                                                                 defaultOpenDir,
                                                                 QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        self.imagePath.setText(imageDirPath)
        print(imageDirPath)

    def openLabelsDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return
        defaultOpenDir = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            labelDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                 '%s - Open Directory' % __appname__,
                                                                 defaultOpenDir,
                                                                 QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        self.labelPath.setText(labelDirPath)
        print(labelDirPath)

    def openClassesDirDialog(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.txt']
        filters = "Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        classFilePath = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if classFilePath:
            if isinstance(classFilePath, (tuple, list)):
                classFilePath = classFilePath[0]
        self.classPath.setText(classFilePath)
        print(classFilePath)

    def getDetailsForTraining(self):
        projectFolderPath=self.projectPath.text()
        imagesFolderPath= self.imagePath.text()
        labelsFolderPath = self.labelPath.text()
        classFilePath=self.classPath.text()
        noOfClasses=self.noOfClassesPath.text()
        noOfClasses=int(noOfClasses)

        zipFilename = 'train_data.zip'

        #projectFolderPath = "E:/projects/data/New folder/images"
        #imagesFolderPath = "E:/projects/data/New folder/images"
        #labelsFolderPath = "E:/projects/data/New folder/labels"
        #classFilePath = "E:/projects/data/classes.txt"
        #noOfClasses = 5


        dataStatus = createTraingdataSet(projectFolderPath,imagesFolderPath,labelsFolderPath,classFilePath,noOfClasses,zipFilename)
        status = dataStatus.getStatus()
        print(status)
        postToRest= postTrainingDetailsToRest(zipFilename)
        postStatusResponse = postToRest.getPostStatus()
        print(postStatusResponse)
    def createGridLayout(self,lastOpenDir):

        self.horizontalGroupBox = QGroupBox("Grid")

        images = []
        path = lastOpenDir
        print(path)
        numberOfrows=0;
        if (path != "" and path != "None"):
            for file in os.listdir(path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    images.append(os.path.join(path, file));
            numberOfrows=len(images)/4
            numberOfrows =int(numberOfrows)+1

            layout = QGridLayout()
            for i in reversed(range(self.tab1.layout.count())):
                self.tab1.layout.itemAt(i).widget().setParent(None)
            k = 0

            self.vbox_choice_img = QHBoxLayout()
            for row in range(numberOfrows):
                for column in range(4):
                    label = QLabel(self)
                    label2 = QLabel()
                    try:
                        try:
                            _fromUtf8 = QtCore.QString.fromUtf8
                        except AttributeError:
                            _fromUtf8 = lambda s: s
                        isTextfileavailable = 0
                        txtFilePath=''
                        if self.defaultSaveDir is not None:
                            basename = os.path.basename(
                                os.path.splitext(images[k])[0])
                            txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)

                            """Annotation file priority:
                            PascalXML > YOLO
                            """
                            if os.path.isfile(txtPath):
                                isTextfileavailable =1
                                txtFilePath=txtPath

                        else:
                            txtPath = os.path.splitext(images[k])[0] + TXT_EXT
                            if os.path.isfile(txtPath):
                                isTextfileavailable =1
                                txtFilePath=txtPath

                        if(isTextfileavailable==0):
                            img = cv2.imread(images[k])
                            height, width, channel = img.shape
                            bytesPerLine = 3 * width
                            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                            button = QPushButton('',self)
                            Icon = QtGui.QIcon()
                            Icon.addPixmap(QtGui.QPixmap(_fromUtf8(qImg)), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                            button.setIcon(Icon)
                            button.setIconSize(QtCore.QSize(200, 200))
                            button.clicked.connect(lambda checked,file=images[k]: self.loadSelectedImage(file))

                            layout.addWidget(button, row, column)
                            k += 1
                        else:
                            img = cv2.imread(images[k])
                            height, width, channel = img.shape
                            bytesPerLine = 3 * width
                            bndBoxFile = open(txtFilePath, 'r')
                            for bndBox in bndBoxFile:
                                classIndex, xcen, ycen, w, h = bndBox.strip().split(' ')
                                xmin, ymin, xmax, ymax = self.yoloLine2ShapeforTabDisplay(xcen, ycen, w, h, height, width)
                                # Draw a diagonal blue line with thickness of 5 px
                                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                            button = QPushButton('', self)
                            Icon = QtGui.QIcon()
                            Icon.addPixmap(QtGui.QPixmap(_fromUtf8(qImg)), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                            button.setIcon(Icon)
                            button.setIconSize(QtCore.QSize(200, 200))

                            # button.clicked.connect(self.clickme)
                            button.clicked.connect(lambda checked, file=images[k]: self.loadSelectedImage(file))

                            layout.addWidget(button, row, column)
                            k += 1
                    except:
                        print("An exception occurred")
            self.imagesRefreshButton = QPushButton("Refresh")
            self.imagesRefreshButton.setFixedHeight(25)
            self.imagesRefreshButton.setFixedWidth(100)
            self.imagesRefreshButton.clicked.connect(lambda checked: self.createGridLayout(self.lastOpenDir))
            self.tab1.layout.addWidget(self.imagesRefreshButton)
            self.tab1.setLayout(self.tab1.layout)
            self.horizontalGroupBox.setLayout(layout)
            scroll = QScrollArea()

            scroll.setWidget(self.horizontalGroupBox)
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(600)
            self.tab1.layout.addWidget(scroll)
            self.tab1.setLayout(self.tab1.layout)

    def yoloLine2ShapeforTabDisplay(self, xcen, ycen, w, h, height, width):

        xmin = max(float(xcen) - float(w) / 2, 0)
        xmax = min(float(xcen) + float(w) / 2, 1)
        ymin = max(float(ycen) - float(h) / 2, 0)
        ymax = min(float(ycen) + float(h) / 2, 1)

        xmin = int(width * xmin)
        xmax = int(width * xmax)
        ymin = int(height * ymin)
        ymax = int(height * ymax)

        return xmin, ymin, xmax, ymax

    def loadSelectedImage(self,file):
        print("file")
        self.loadFile(file)

    def autoUploadandTrain(self):
        if(checkfortrain >=3):
            self.AutouploadButton.setEnabled(False)
            self.autotrainUpload()
        else:
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#ff0000;\" >"
            redText += "Please annotate atleast two images"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)



    def autotrainUpload(self):
        if checkfortrain >= 2:
            self.autoTrainStatus=''
            print("creating data")
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#000000;\" >"
            redText += "creating data"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)

            file1 = open('annotatedData.txt', 'r')
            count = 0
            if not os.path.exists('train_data'):
                os.mkdir('train_data')
            else:
                shutil.rmtree("train_data")
                os.mkdir('train_data')

            dest_path = os.path.abspath(os.getcwd())
            if not os.path.exists(os.path.join(dest_path + "/train_data", "images")):
                os.makedirs(os.path.join(dest_path + "/train_data", "images"))
                os.makedirs(os.path.join(dest_path + "/train_data", "labels"))

            while True:
                count += 1
                line = file1.readline()
                if not line:
                    break
                path = line.strip()
                if (path.endswith('.txt')):

                    filename = Path(path).name
                    filename_wo_ext, file_extension = os.path.splitext(filename)
                    j = 0
                    for j in range(2):
                        labeltarget = r'' + dest_path + '/train_data/labels/' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelaveraging_target = r'' + dest_path + '/train_data/labels/averaging_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelgausian_target = r'' + dest_path + '/train_data/labels/gausian_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelbilateral_target = r'' + dest_path + '/train_data/labels/bilateral_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        labelmedian_target = r'' + dest_path + '/train_data/labels/median_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        target = r'' + dest_path + '/train_data/images/' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        averaging_target = r'' + dest_path + '/train_data/images/averaging_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        gausian_target = r'' + dest_path + '/train_data/images/gausian_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        bilateral_target = r'' + dest_path + '/train_data/images/bilateral_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        median_target = r'' + dest_path + '/train_data/images/median_' + filename_wo_ext + "_" + str(
                            j) + file_extension
                        shutil.copyfile(path, target)
                        shutil.copyfile(path, averaging_target)
                        shutil.copyfile(path, gausian_target)
                        shutil.copyfile(path, bilateral_target)
                        shutil.copyfile(path, median_target)
                        shutil.copyfile(path, labeltarget)
                        shutil.copyfile(path, labelaveraging_target)
                        shutil.copyfile(path, labelgausian_target)
                        shutil.copyfile(path, labelbilateral_target)
                        shutil.copyfile(path, labelmedian_target)
                        j += 1
                else:
                    try:
                        # print(path)
                        img = cv2.imread(path)
                        averaging = cv2.blur(img, (5, 5))
                        gaussian = cv2.GaussianBlur(img, (5, 5), 1)
                        median = cv2.medianBlur(img, 5)
                        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
                        filename = Path(path).name
                        filename_wo_ext, file_extension = os.path.splitext(filename)
                        dest_path = os.path.abspath(os.getcwd())
                        if not os.path.exists(os.path.join(dest_path + "/train_data", "images")):
                            os.makedirs(os.path.join(dest_path + "/train_data", "images"))
                        j = 0
                        for j in range(2):
                            cv2.imwrite(dest_path + '/train_data/images/' + filename_wo_ext + "_" + str(
                                j) + file_extension, img)
                            cv2.imwrite(dest_path + '/train_data/images/averaging_' + filename_wo_ext + "_" + str(
                                j) + file_extension, averaging)
                            cv2.imwrite(dest_path + '/train_data/images/gausian_' + filename_wo_ext + "_" + str(
                                j) + file_extension, gaussian)
                            cv2.imwrite(dest_path + '/train_data/images/bilateral_' + filename_wo_ext + "_" + str(
                                j) + file_extension, bilateral)
                            cv2.imwrite(dest_path + '/train_data/images/median_' + filename_wo_ext + "_" + str(
                                j) + file_extension, median)
                            j += 1
                    except:
                        print("This is an error message!")

            file1.close()
            # create .names file
            class_path = os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'
            # print("classpath")
            # print(class_path)
            names_path = os.path.abspath(os.getcwd()) + '/train_data/train_data.names'
            shutil.copyfile(class_path, names_path)

            # create .data file
            self.createDataFileAutoTrain()

            # make change in the config file
            self.changeYoloCfgFileAutoTrain()
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#000000;\" >"
            redText += "Succesfully created the train.data, train.names and yolo.cfg"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)



            zipFilename = "train_data.zip"
            shutil.make_archive('train_data', 'zip', 'train_data')
            print("successfully created the data")
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#00580a;\" >"
            redText += "Successfully created the train_data.zip file"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)
            self.upload_to_awsAutoTrain(zipFilename, 'testbreezbucket', 'train_data.zip')
            if (self.autoTrainStatus == 'True'):
                self.postAutoTrainingDetailsToRestwithopencvdata(zipFilename)

    def createDataFileAutoTrain(self):
        num_lines = sum(1 for line in open(os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'))
        with open(os.path.abspath(os.getcwd()) + '/train_data/train.data', 'a') as the_file:
            the_file.write('classes= ' + str(num_lines) + '\n')
            the_file.write('train  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/train.txt') + '\n')
            the_file.write('valid  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/valid.txt') + '\n')
            the_file.write('names  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/training.names') + '\n')
            the_file.write('backup  = ' + str(os.path.abspath(os.getcwd()) + '/train_data/backup/') + '\n')
            the_file.write('eval=coco' + '\n')


    def changeYoloCfgFileAutoTrain(self):
        numOfClasses = sum(1 for line in open(os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'))
        filters = ((numOfClasses + 5) * 3)

        a_file = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "r")
        list_of_lines = a_file.readlines()
        list_of_lines[609] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[695] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[782] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[602] = "filters=" + str(filters) + '\n'
        list_of_lines[688] = "filters=" + str(filters) + '\n'
        list_of_lines[775] = "filters=" + str(filters) + '\n'

        a_file = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        shutil.copyfile(os.path.abspath(os.getcwd()) + '/yolov3.cfg', os.path.abspath(os.getcwd()) + '/train_data/yolov3.cfg')

    def upload_to_awsAutoTrain(self,zipFilename, bucket, s3_file):
        redText = "<span style=\" font-size:9pt; font-weight:500; color:#000000;\" >"
        redText += "Uploading zip to S3"
        redText += "</span>"
        self.autoAnnotateOutput.append(redText)
        self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
        self.tabnav4.setLayout(self.autoannotateDetailsbox)
        self.ACCESS_KEY = ' '
        self.SECRET_KEY = ' '
        print("Uploading to S3")
        s3 = boto3.client('s3', aws_access_key_id=self.ACCESS_KEY,
                          aws_secret_access_key=self.SECRET_KEY)

        try:
            s3.upload_file(zipFilename, bucket, s3_file, ExtraArgs={'ACL':'public-read'})
            print("Upload Successful to S3")
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#00580a;\" >"
            redText += "Successfully Uploaded zip file to S3"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)
            self.autoTrainStatus = 'True'
            return True
        except FileNotFoundError:
            self.autoTrainStatus = 'false'
            print("The file was not found")
            return False

    def postAutoTrainingDetailsToRestwithopencvdata(self,zipFilename):
        uniqueUuid = uuid.uuid1()
        data={
                "requestDetails": {
                "action": "START_TRAIN",
                "uuid": str(uniqueUuid),
                "fileName": zipFilename,
                "archiveLink": "https://testbreezbucket.s3.amazonaws.com/train_data.zip"
                },
                "deviceCode": "ABCD_2"
                }
        data_json = json.dumps(data)
        headers = {'Content-type': 'application/json'}
        url = 'http://192.168.1.5:8080/vflow/upload_high_priority_req_res_msgs'
        try:
            response = requests.post(url, data=data_json, headers=headers)
            self.status = "Successfully posted to endpoint"
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#00580a;\" >"
            redText += "Successfully Posted to end point"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)
        except:
            self.status = response
            print("An exception occurred")

    def getAutoTrainWeights(self):
        global previousDownloadedWeightFile
        resp = requests.get('http://192.168.1.5:8080/vflow/download_high_priority_req_res_msgs?device_code=ABCD_1')
        data=[{}]
        logdata = [{}]

        AutotrainweightUrl = []
        Autotrainweightname =[]
        if (resp.status_code == 200):
            try:
                data = resp.json()
                logdata = resp.json()
                print(data)
            except:
                print("Empty json")
        logs = ''
        autologsdata = [{}]
        try:
            for element in logdata:
                element.pop('weights', None)
                autologsdata = logdata
        except:
            print("Empty logs")
        try:
            logslist = {k: v for d in autologsdata for k, v in d.items()}
            logs = logslist['logs']
        except:
            print("Empty logs")


        if (logs != ''):
            font = self.logOutput.font()
            font.setFamily("Courier")
            font.setPointSize(15)
            self.logOutput.append(logs)

            self.logDetailsbox.addWidget(self.logOutput)
            self.tabnav3.setLayout(self.logDetailsbox)
        try:
            for items in data:
                for item in items["weights"]:
                    if (item["name"] != ''):
                        Autotrainweightname.append(item['name'])
                        AutotrainweightUrl.append(item['url'])
        except:
            print("Empty weights")
        if (len(AutotrainweightUrl) != 0):
            url=AutotrainweightUrl[-1]
            filename=Autotrainweightname[-1]
            if(filename != previousDownloadedWeightFile):
                previousDownloadedWeightFile =filename
                output_path = "yolov3/cfg/"+filename
                with DownloadProgressBar(unit='B', unit_scale=True,
                                         miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
                from yolov3.detect import yolodetection
                self.imagefolder=self.lastOpenDir
                self.labelpath=self.defaultSaveDir
                print(self.imagefolder)
                print(self.labelpath)
                yolodetection(filename,self.imagefolder,self.labelpath)
                redText = "<span style=\" font-size:9pt; font-weight:500; color:#00580a;\" >"
                redText += "Successfully ran detection"
                redText += "</span>"
                self.autoAnnotateOutput.append(redText)
                self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
                self.tabnav4.setLayout(self.autoannotateDetailsbox)
            else:
                print("No new Weight File Generated")
                redText = "<span style=\" font-size:9pt; font-weight:500; color:#00580a;\" >"
                redText += "No new Weight File Generated"
                redText += "</span>"
                self.autoAnnotateOutput.append(redText)
                self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
                self.tabnav4.setLayout(self.autoannotateDetailsbox)
        else:
            print("No new Weight File Generated")
            redText = "<span style=\" font-size:9pt; font-weight:500; color:#00580a;\" >"
            redText += "No new Weight File Generated"
            redText += "</span>"
            self.autoAnnotateOutput.append(redText)
            self.autoannotateDetailsbox.addWidget(self.autoAnnotateOutput)
            self.tabnav4.setLayout(self.autoannotateDetailsbox)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.usingPascalVocFormat = True
            self.usingYoloFormat = False
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.usingPascalVocFormat = False
            self.usingYoloFormat = True
            LabelFile.suffix = TXT_EXT

    def change_format(self):
        if self.usingPascalVocFormat:
            self.set_format(FORMAT_YOLO)
        elif self.usingYoloFormat:
            self.set_format(FORMAT_PASCALVOC)

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner() \
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)

        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    def fileitemDoubleClicked1(self, item=None):
       print("INSIDE fileitemDoubleClicked")

    # Add chris
    def btnstate(self, item=None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item:  # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count() - 1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]
        self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape)
        self.updateComboBox()
        self.canvas.loadShapes(s)

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                        # add chris
                        difficult=s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.usingYoloFormat is True:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                                              self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == "":
                self.labelList.item(i).setCheckState(2)
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
                print("load Image file")
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            # if self.usingPascalVocFormat is True:
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(self.filePath)[0])
                xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)

                """Annotation file priority:
                PascalXML > YOLO
                """
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)
            else:
                xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                txtPath = os.path.splitext(filePath)[0] + TXT_EXT
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull() \
                and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def on_timeout(self):
        try:
            file = next(self.files_it)
            pixmap = QtGui.QPixmap(file)
            self.add_pixmap(pixmap)
        except StopIteration:
            self._timer.stop()

    def add_pixmap(self, pixmap):
        if not pixmap.isNull():
            label = QtWidgets.QLabel(pixmap=pixmap)
            self._lay.addWidget(label)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                        '%s - Save annotations to the directory' % __appname__, path,
                                                        QFileDialog.ShowDirsOnly
                                                        | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath)) \
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self, '%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True:
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                                  '%s - Open Directory' % __appname__,
                                                                  defaultOpenDirPath,
                                                                  QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        print("********")
        print(defaultOpenDirPath)
        print(targetDirPath)
        if (defaultOpenDirPath == targetDirPath and len(weightName) == 0):
            f = open('previous_conf.txt')
            lines = f.read().split("\n")
            for i in range(len(lines)):
                print(lines[i])
                if(lines[i] != ""):
                    x ,y = lines[i].split(",")
                    weightName.append(x)
                    weightUrl.append(y)
            f.close()
            if (len(weightName) != 0):

                self.table.setColumnCount(2)
                self.table.setRowCount(len(weightName))

                for index in range(len(weightName)):
                    item1 = QTableWidgetItem(weightName[index])
                    self.table.setItem(index, 0, item1)
                    self.btn_sell = QPushButton('Download')
                    self.btn_sell.clicked.connect(
                        lambda checked, file=weightUrl[index]: self.downloadSelectedWeight(file))
                    self.table.setCellWidget(index, 1, self.btn_sell)

            self.weightsbox.addWidget(self.table)
            self.tabnav2.setLayout(self.weightsbox)

        if(targetDirPath != ""):
            defaultOpenDirPath =targetDirPath

        print("&&&&&&")
        print(defaultOpenDirPath)
        print(targetDirPath)
        self.lastOpenDir = targetDirPath
        self.importDirImages(targetDirPath)
        self.createGridLayout(targetDirPath)


    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        #self.loadFile()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)



    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
                #print("*****filename********")
                #print(filename)


        if filename:
            self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        global checkfortrain
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileDir = os.path.dirname(self.filePath)
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                self._saveFile(savedPath)
                checkfortrain += 1
                Imagepath = str(imgFileDir) + "/" + str(savedFileName) + ".jpg"
                Imagepath = Imagepath.replace("\\", "/")
                print("Imagepath")
                print(Imagepath)
                Labelpath = str(savedPath) + ".txt"
                Labelpath = Labelpath.replace("\\", "/")
                print(Labelpath)
                with open('annotatedData.txt', 'a') as the_file:
                    the_file.write(str(Imagepath)+'\n')
                    the_file.write(str(Labelpath)+'\n')
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            checkfortrain += 1
            Imagepath = str(imgFileDir) + "/" + str(savedFileName) + ".jpg"
            Imagepath = Imagepath.replace("\\", "/")
            #print(Imagepath)
            Labelpath = str(savedPath) + ".txt"
            Labelpath = Labelpath.replace("\\", "/")
            #print(Labelpath)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

        #if (checkfortrain >= 2):
            #ret = QMessageBox.question(self, 'MessageBox', "Creating data and uploading for training",
            #                           QMessageBox.Ok)

            #if ret == QMessageBox.Ok:
                #autoUpload=startTrainAutomatically(checkfortrain)
                #postStatusResponse = autoUpload.getStatus()
                #print(postStatusResponse)



    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def annotate(self, _value=False):
        if checkfortrain > 2:
            print(checkfortrain)
            file1 = open('annotatedData.txt', 'r')
            count = 0
            while True:
                count += 1
                line = file1.readline()
                if not line:
                    break
                path=line.strip()
                if (path.endswith('.txt')):
                    if not os.path.exists('darknet_training'):
                        os.makedirs('darknet_training')
                    dest_path=os.path.abspath(os.getcwd())
                    if not os.path.exists(os.path.join(dest_path+"/darknet_training", "images")):
                        os.makedirs(os.path.join(dest_path+"/darknet_training", "images"))

                    filename=Path(path).name
                    filename_wo_ext, file_extension = os.path.splitext(filename)
                    j = 0
                    for j in range(40):
                        target = r''+dest_path+'/darknet_training/images/' + filename_wo_ext + "_" + str(j) + file_extension
                        averaging_target = r''+dest_path+'/darknet_training/images/averaging_' + filename_wo_ext + "_" + str(j) + file_extension
                        gausian_target = r''+dest_path+'/darknet_training/images/gausian_' + filename_wo_ext + "_" + str(j) + file_extension
                        bilateral_target = r''+dest_path+'/darknet_training/images/bilateral_' + filename_wo_ext + "_" + str(j) + file_extension
                        median_target = r''+dest_path+'/darknet_training/images/median_' + filename_wo_ext + "_" + str(j) + file_extension
                        shutil.copyfile(path, target)
                        shutil.copyfile(path, averaging_target)
                        shutil.copyfile(path, gausian_target)
                        shutil.copyfile(path, bilateral_target)
                        shutil.copyfile(path, median_target)
                        j += 1
                else:
                    try:
                        print(path)
                        img = cv2.imread(path)
                        averaging = cv2.blur(img, (5, 5))
                        gaussian = cv2.GaussianBlur(img, (5, 5), 1)
                        median = cv2.medianBlur(img, 5)
                        bilateral = cv2.bilateralFilter(img, 9, 75, 75)
                        filename = Path(path).name
                        filename_wo_ext, file_extension = os.path.splitext(filename)
                        dest_path = os.path.abspath(os.getcwd())
                        if not os.path.exists(os.path.join(dest_path + "/darknet_training", "images")):
                            os.makedirs(os.path.join(dest_path + "/darknet_training", "images"))
                        j = 0
                        for j in range(40):
                            cv2.imwrite(dest_path + '/darknet_training/images/' + filename_wo_ext + "_" + str(
                                j) + file_extension, img)
                            cv2.imwrite(dest_path + '/darknet_training/images/averaging_' + filename_wo_ext + "_" + str(
                                j) + file_extension, averaging)
                            cv2.imwrite(dest_path + '/darknet_training/images/gausian_' + filename_wo_ext + "_" + str(
                                j) + file_extension, gaussian)
                            cv2.imwrite(dest_path + '/darknet_training/images/bilateral_' + filename_wo_ext + "_" + str(
                                j) + file_extension, bilateral)
                            cv2.imwrite(dest_path + '/darknet_training/images/median_' + filename_wo_ext + "_" + str(
                                j) + file_extension, median)
                            j += 1
                    except:
                        print("This is an error message!")

            file1.close()
            #create .names file
            class_path = os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'
            print("classpath")
            print(class_path)
            names_path = os.path.abspath(os.getcwd()) + '/darknet_training/training.names'
            shutil.copyfile(class_path, names_path)

            #create .data file
            self.createDataFile()

            # create train and test files
            self.createTestandTrainFiles()

            #make change in the config file
            self.changeYoloCfgFile()

            #make change in the config file
            self.startTraining()

        else:
            if self.noannotatedimages():
                return

    def createTestandTrainFiles(self):
        annotation_dir = os.path.abspath(os.getcwd())+'/darknet_training/images'
        percentage = 10
        percentage_test = percentage
        # Create and/or truncate train.txt and test.txt
        file_train = open(os.path.abspath(os.getcwd())+'/darknet_training/train.txt', 'w')
        file_test = open(os.path.abspath(os.getcwd())+'/darknet_training/test.txt', 'w')
        # Populate train.txt and test.txt
        counter = 1
        index_test = round(100 / percentage_test)
        for pathAndFilename in glob.iglob(os.path.join(annotation_dir, "*.jpg")):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))
            if counter == index_test:
                counter = 1
                file_test.write(annotation_dir + "/" + title + '.jpg' + "\n")
            else:
                file_train.write(annotation_dir + "/" + title + '.jpg' + "\n")
                counter = counter + 1

    def createDataFile(self):
        num_lines = sum(1 for line in open(os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'))
        with open(os.path.abspath(os.getcwd()) + '/darknet_training/training.data', 'a') as the_file:
            the_file.write('classes= ' + str(num_lines) + '\n')
            the_file.write('train  = ' + str(os.path.abspath(os.getcwd()) + '/darknet_training/train.txt') + '\n')
            the_file.write('valid  = ' + str(os.path.abspath(os.getcwd()) + '/darknet_training/valid.txt') + '\n')
            the_file.write('names  = ' + str(os.path.abspath(os.getcwd()) + '/darknet_training/training.names') + '\n')
            the_file.write('backup  = ' + str(os.path.abspath(os.getcwd()) + '/darknet_training/backup/') + '\n')
            the_file.write('eval=coco' + '\n')


    def changeYoloCfgFile(self):
        numOfClasses = sum(1 for line in open(os.path.abspath(os.getcwd()) + '/data/predefined_classes.txt'))
        filters = ((numOfClasses + 5) * 3)

        a_file = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "r")
        list_of_lines = a_file.readlines()
        list_of_lines[609] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[695] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[782] = "classes=" + str(numOfClasses) + '\n'
        list_of_lines[602] = "filters=" + str(filters) + '\n'
        list_of_lines[688] = "filters=" + str(filters) + '\n'
        list_of_lines[775] = "filters=" + str(filters) + '\n'

        a_file = open(os.path.abspath(os.getcwd()) + '/yolov3.cfg', "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        shutil.copyfile(os.path.abspath(os.getcwd()) + '/yolov3.cfg', os.path.abspath(os.getcwd()) + '/darknet_training/yolov3.cfg')

    def startTraining(self):
        os.system('start cmd /k e:/darknet-master/darknet.exe detector train '
                  + os.path.abspath(os.getcwd()) + '/darknet_training/training.data ' +
                  os.path.abspath(os.getcwd()) + '/darknet_training/yolov3.cfg e:/darknet-master/yolov4.conv.137')

    def noannotatedimages(self):
        return not (self.discardChangesDialog1())

    def discardChangesDialog1(self):
        ok = QMessageBox.Close
        msg = u'Please annotate atleast two images to start training'
        return ok == QMessageBox.warning(self, u'Attention', msg, ok)

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            self.defaultSaveDir = os.path.dirname(os.path.abspath(fullFilePath))

            if removeExt:
                return os.path.splitext(fullFilePath)[0]  # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def deleteImg(self):
        deletePath = self.filePath
        if deletePath is not None:
            self.openNextImg()
            os.remove(deletePath)
            self.importDirImages(self.lastOpenDir)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadDefaultClassName(self,predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            f = open(predefClassesFile, 'r')
            line = f.readline()
            f.close()
            return line

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        print(shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)



def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'predefined_classes.txt'),
                     argv[3] if len(argv) >= 4 else None)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    f = open("annotatedData.txt", "w+")
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
