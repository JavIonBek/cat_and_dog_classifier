import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PyQt5 import QtCore, QtGui, QtWidgets
from ui import Ui_MainWindow


# Create application
app = QtWidgets.QApplication(sys.argv)

# Dark style
app.setStyle('Fusion')
palette = QtGui.QPalette()
palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53,53,53))
palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15,15,15))
palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53,53,53))
palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53,53,53))
palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
     
palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142,45,197).lighter())
palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
app.setPalette(palette)

# Create form and init UI
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

# Hook logic
# To prevent TensorFlow from making an error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filePath = []

def setImage():
    '''
    Loading images
    '''
    global filePath
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Load image", "", "Image Files (*.png *.jpg *jpeg *.bmp)") # Ask for file
    if fileName:
        filePath.append(fileName)
        pixmap = QtGui.QPixmap(fileName)
        ui.label.setPixmap(pixmap)
        ui.label.setAlignment(QtCore.Qt.AlignCenter)

def predictImage():
    '''
    Image prediction
    '''
    model = tf.keras.models.load_model("model.h5")
    path = os.path.join(filePath[-1])
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = model.predict(images)
    if prediction[0] > 0.5:
        ui.label_2.setText('<b>Dog</b>')
    else:
        ui.label_2.setText('<b>Cat</b>')


ui.pushButton.clicked.connect(setImage)
ui.pushButton_2.clicked.connect(predictImage)

# Run main loop
sys.exit(app.exec_())
