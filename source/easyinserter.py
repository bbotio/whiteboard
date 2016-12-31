import math
from PyQt4.QtGui import *
from PyQt4.Qt import *
from sloth.items.inserters import RectItemInserter


class EasyRectItemInserter(RectItemInserter):
    """Label by single mouse click, not by holding mouse all time"""
    def __init__(self, labeltool, scene, default_properties=None,
                 prefix="", commit=True):
        super(EasyRectItemInserter, self).__init__(labeltool, scene,
                                                   default_properties, prefix,
                                                   commit)

    def mousePressEvent(self, event, image_item):
        if self._item is not None:
            self._finish(event, image_item)
        else:
            super(EasyRectItemInserter, self).mousePressEvent(event, image_item)

    def keyPressEvent(self, event, image_item):
        self._finish(event, image_item)

    def mouseReleaseEvent(self, event, image_item):
        pass

    def _finish(self, event, image_item):
        if self._item is not None:
            if self._item.rect().width() > 1 and \
               self._item.rect().height() > 1:
                rect = self._item.rect()
                self._ann.update({self._prefix + 'x': rect.x(),
                                  self._prefix + 'y': rect.y(),
                                  self._prefix + 'width': rect.width(),
                                  self._prefix + 'height': rect.height()})
                self._ann.update(self._default_properties)
                if self._commit:
                    image_item.addAnnotation(self._ann)
            self._scene.removeItem(self._item)
            self.annotationFinished.emit()
            self._init_pos = None
            self._item = None

        self._aiming = True
        self._scene.views()[0].viewport().setCursor(Qt.CrossCursor)
        event.accept()
