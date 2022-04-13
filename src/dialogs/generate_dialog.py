
from PyQt5.QtWidgets import QMessageBox

def generate_messagebox(object, title, message):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(title)
    msg.setInformativeText(message)
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Cancel)
    cancel_button = msg.button(QMessageBox.Cancel)
    cancel_button.setText('확안하였습니다')
    return msg.exec_()
    # error_msg.question(object, 'error message','데이터를 먼저 불러와주세요.', QMessageBox.Yes)
