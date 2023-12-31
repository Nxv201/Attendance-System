# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                            QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QPushButton, QSizePolicy, QTabWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)


class Ui_Main(object):
    def setupUi(self, Main):
        if not Main.objectName():
            Main.setObjectName(u"Main")
        Main.resize(500, 500)
        Main.setMinimumSize(QSize(500, 500))
        Main.setMaximumSize(QSize(500, 500))
        self.centralwidget = QWidget(Main)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setStyleSheet(u"QWidget#centralwidget{\n"
                                         "	border-radius: 20px;\n"
                                         "}\n"
                                         "QWidget#MainTab{\n"
                                         "	background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0.023, stop:0 rgba(255, 140, 140, 255), stop:1 rgba(94, 122, 255, 255));\n"
                                         "}\n"
                                         "QWidget#AttendanceTab{\n"
                                         "	background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0.023, stop:0 rgba(255, 140, 140, 255), stop:1 rgba(94, 122, 255, 255));\n"
                                         "}\n"
                                         "QWidget#RegisterTab{\n"
                                         "	background-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0.023, stop:0 rgba(255, 140, 140, 255), stop:1 rgba(94, 122, 255, 255));\n"
                                         "}\n"
                                         "QFrame#frame{\n"
                                         "	margin: 60px;\n"
                                         "	border-radius: 20px;\n"
                                         "	background-color: rgba(225, 228, 221, 120)\n"
                                         "}\n"
                                         "\n"
                                         "QLineEdit{\n"
                                         "	min-height: 45px;\n"
                                         "	border-radius: 20px;\n"
                                         "	background-color: #FFFFFF;\n"
                                         "	padding-left: 20px;\n"
                                         "	color: rgb(140, 140, 140);\n"
                                         "}\n"
                                         "\n"
                                         "QLineEdit:hover{\n"
                                         "	border: 2px solid rgb(139, 142, 139);\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#attend_button{\n"
                                         "	min-height: 45px;\n"
                                         "	border-radius: 20px;\n"
                                         ""
                                         "	background-color: rgb(140, 140, 140);\n"
                                         "	color: #FFFFFF;\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#attend_button:hover{\n"
                                         "	border: 2px solid rgb(255, 255, 255);\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#add_button{\n"
                                         "	min-height: 45px;\n"
                                         "	border-radius: 20px;\n"
                                         "	background-color: rgb(140, 140, 140);\n"
                                         "	color: #FFFFFF;\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#add_button:hover{\n"
                                         "	border: 2px solid rgb(255, 255, 255);\n"
                                         "}\n"
                                         "\n"
                                         "QCheckBox{\n"
                                         "	font-size: 10px;\n"
                                         "	color: #FFFFFF;\n"
                                         "}\n"
                                         "\n"
                                         "QLabel{\n"
                                         "	color: rgb(95, 94, 108);\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#forgot_button{\n"
                                         "	border: 0px;\n"
                                         "	font-style: italic;\n"
                                         "	font-size: 10px;\n"
                                         "	color: #FFFFFF;\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#close_button{\n"
                                         "	background-color: rgb(186, 0, 0);\n"
                                         "	border-radius: 6px;\n"
                                         "}\n"
                                         "\n"
                                         "QPushButton#minimize_button{\n"
                                         "	background-color: rgb(226, 226, 0);\n"
                                         "	border-radius: 6px;\n"
                                         "}\n"
                                         "")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setStyleSheet(u"")
        self.MainTab = QWidget()
        self.MainTab.setObjectName(u"MainTab")
        self.verticalLayout = QVBoxLayout(self.MainTab)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(30, 0, 30, 0)
        self.label_2 = QLabel(self.MainTab)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMaximumSize(QSize(16777215, 40))
        self.label_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_2)

        self.lineEdit = QLineEdit(self.MainTab)
        self.lineEdit.setObjectName(u"lineEdit")

        self.verticalLayout.addWidget(self.lineEdit)

        self.lineEdit_2 = QLineEdit(self.MainTab)
        self.lineEdit_2.setObjectName(u"lineEdit_2")

        self.verticalLayout.addWidget(self.lineEdit_2)

        self.add_button = QPushButton(self.MainTab)
        self.add_button.setObjectName(u"add_button")

        self.verticalLayout.addWidget(self.add_button)

        self.attend_button = QPushButton(self.MainTab)
        self.attend_button.setObjectName(u"attend_button")

        self.verticalLayout.addWidget(self.attend_button)

        self.tabWidget.addTab(self.MainTab, "")
        self.RegisterTab = QWidget()
        self.RegisterTab.setObjectName(u"RegisterTab")
        self.label_3 = QLabel(self.RegisterTab)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(30, 30, 436, 40))
        self.label_3.setMaximumSize(QSize(16777215, 40))
        self.label_3.setAlignment(Qt.AlignCenter)
        self.registerTable = QTableWidget(self.RegisterTab)
        if (self.registerTable.columnCount() < 3):
            self.registerTable.setColumnCount(3)
        __qtablewidgetitem = QTableWidgetItem()
        self.registerTable.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.registerTable.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.registerTable.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        self.registerTable.setObjectName(u"registerTable")
        self.registerTable.setGeometry(QRect(0, 120, 501, 351))
        self.tabWidget.addTab(self.RegisterTab, "")
        self.AttendanceTab = QWidget()
        self.AttendanceTab.setObjectName(u"AttendanceTab")
        self.label_4 = QLabel(self.AttendanceTab)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(30, 30, 436, 40))
        self.label_4.setMaximumSize(QSize(16777215, 40))
        self.label_4.setAlignment(Qt.AlignCenter)
        self.attendTable = QTableWidget(self.AttendanceTab)
        if (self.attendTable.columnCount() < 3):
            self.attendTable.setColumnCount(3)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.attendTable.setHorizontalHeaderItem(0, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.attendTable.setHorizontalHeaderItem(1, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.attendTable.setHorizontalHeaderItem(2, __qtablewidgetitem5)
        self.attendTable.setObjectName(u"attendTable")
        self.attendTable.setGeometry(QRect(0, 120, 500, 351))
        self.tabWidget.addTab(self.AttendanceTab, "")

        self.verticalLayout_2.addWidget(self.tabWidget)

        Main.setCentralWidget(self.centralwidget)

        self.retranslateUi(Main)

        self.tabWidget.setCurrentIndex(2)

        QMetaObject.connectSlotsByName(Main)

    # setupUi

    def retranslateUi(self, Main):
        Main.setWindowTitle(QCoreApplication.translate("Main", u"MainWindow", None))
        self.label_2.setText(QCoreApplication.translate("Main",
                                                        u"<html><head/><body><p><span style=\" font-size:20pt;\">Sign In</span></p></body></html>",
                                                        None))
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("Main", u"Email", None))
        self.lineEdit_2.setPlaceholderText(QCoreApplication.translate("Main", u"User name", None))
        self.add_button.setText(QCoreApplication.translate("Main", u"Add User", None))
        self.attend_button.setText(QCoreApplication.translate("Main", u"Take Attendance", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.MainTab),
                                  QCoreApplication.translate("Main", u"Main", None))
        self.label_3.setText(QCoreApplication.translate("Main",
                                                        u"<html><head/><body><p><span style=\" font-size:20pt;\">Register Table</span></p><p><br/></p></body></html>",
                                                        None))
        ___qtablewidgetitem = self.registerTable.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("Main", u"ID", None));
        ___qtablewidgetitem1 = self.registerTable.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("Main", u"NAME", None));
        ___qtablewidgetitem2 = self.registerTable.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("Main", u"EMAIL", None));
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.RegisterTab),
                                  QCoreApplication.translate("Main", u"Register", None))
        self.label_4.setText(QCoreApplication.translate("Main",
                                                        u"<html><head/><body><p><span style=\" font-size:20pt;\">Attendance Table</span></p><p><br/></p></body></html>",
                                                        None))
        ___qtablewidgetitem3 = self.attendTable.horizontalHeaderItem(0)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("Main", u"ID", None));
        ___qtablewidgetitem4 = self.attendTable.horizontalHeaderItem(1)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("Main", u"NAME", None));
        ___qtablewidgetitem5 = self.attendTable.horizontalHeaderItem(2)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("Main", u"TIME", None));
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.AttendanceTab),
                                  QCoreApplication.translate("Main", u"Attendance", None))
    # retranslateUi
