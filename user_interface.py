from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from AIUtil import home, add, GrandData, start, homeUser, del_user, edit_user



class UserInterface(QtCore.QObject): #An object wrapping around our ui
    attendData: GrandData
    userData: GrandData
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui_main.ui")

        self.ui.registerTable.setColumnWidth(0, 100)
        self.ui.registerTable.setColumnWidth(1, 200)
        self.ui.registerTable.setColumnWidth(2, 200)
        self.ui.add_button.clicked.connect(self.addUser)
        self.ui.attend_button.clicked.connect(self.takeAttendence)
        self.ui.del_button.clicked.connect(self.delUser)
        self.ui.edit_button.clicked.connect(self.editUser)
        self.attendData = home()
        self.userData = homeUser()
        self.loadAttendData()
        self.loadUserData()

    def show(self):
        self.ui.show()

    def addUser(self):
        id = self.ui.user_id.text()
        name = self.ui.user_name.text()
        if id != "" and name != "":
            self.userData = add(name, id)
            #self.attendData = home()
            self.loadUserData()
            self.ui.user_id.clear()
            self.ui.user_name.clear()
        else: 
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please enter information!")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

            # Show the error dialog
            msg_box.exec()

    def takeAttendence(self):
        self.attendData = start()
        self.loadAttendData()

    def delUser(self):
        id = self.ui.user_id_for_del.text()
        self.ui.user_id_for_del.clear()
        if id not in self.userData.rolls.values:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("ID không tồn tại.")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()  # Show the error dialog
        else:
            reply = QMessageBox.question(None, 'Confirmation', 'Bạn có muốn tiếp tục?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                ret = del_user(id)
                self.userData = homeUser()
                self.loadUserData()
                if ret == 0:
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Information")
                    msg_box.setText("Xoá thành công!")
                    msg_box.setStandardButtons(QMessageBox.Ok)
                    msg_box.exec_()
            else:
                pass

    def editUser(self):
        id = self.ui.user_id_for_edit.text()
        new_username = self.ui.new_user_name.text()
        self.ui.user_id_for_edit.clear()
        self.ui.new_user_name.clear()
        if id not in self.userData.rolls.values:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("ID không tồn tại.")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()  # Show the error dialog
        else:
            ret = edit_user(userid=id, new_username=new_username)
            self.userData = homeUser()
            self.loadUserData()
            if ret == 0:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setWindowTitle("Information")
                msg_box.setText("Sửa thành công!")
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec_()

    def loadUserData(self):

        row=0
        if self.userData.l > 0:
            self.ui.registerTable.setRowCount(self.userData.l)
            for i in range(self.userData.l):
                item = QtWidgets.QTableWidgetItem(str(self.userData.rolls[i]))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.ui.registerTable.setItem(row, 0, item)

                item = QtWidgets.QTableWidgetItem(str(self.userData.names[i]))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.ui.registerTable.setItem(row, 1, item)
                row += 1

    def loadAttendData(self):
        row=0
        if self.attendData.l > 0:
            self.ui.attendTable.setRowCount(self.attendData.l)
            for i in range(self.attendData.l):
                item = QtWidgets.QTableWidgetItem(str(self.attendData.rolls[i]))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.ui.attendTable.setItem(row, 0, item)

                item = QtWidgets.QTableWidgetItem(str(self.attendData.names[i]))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.ui.attendTable.setItem(row, 1, item)

                item = QtWidgets.QTableWidgetItem(str(self.attendData.times[i]))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.ui.attendTable.setItem(row, 2, item)
                row += 1
