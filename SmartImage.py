import numpy as np
import matplotlib.pyplot as plt

class SmartImage:
    def __init__(self, image, coordinates):
        self.origin = None
        self.label = None
        self.img = image
        self.coord = coordinates
    
    def contains(self, point):
        assert len(point) == 2
        x, y = point[0], point[1]

        xs = self.coord[:,1]
        x_high, x_low = xs.max(), xs.min()
        ys = self.coord[:,0]
        y_high, y_low = ys.max(), ys.min()
        
        if x_low <= x <= x_high and y_low <= y <= y_high:
            return True
        else:
            return False

    def setLabel(self, label):
        self.label = label

    def setOrigin(self, origin):
        self.origin = origin

    def fliplr(self):
        assert isinstance(self.img, np.ndarray), "Image is not an numpy.ndarray instance."
        # 0,10
                # 50,50
                
        # 50,10         0,10
        # 50,50         0, 50
        self.img = np.fliplr(self.img)
        
        self.coord[0,0], self.coord[1,0] = self.coord[1,0], self.coord[0,0] 

    def rot90(self, number = 1):
        assert isinstance(self.img, np.ndarray), "Image is not an numpy.ndarray instance."
        # 0,10       50,10
        # 0,50       50,50
        
        # 0,50       0,10
        # 50,50       50,10
                
        for i in range(number):
            self.img = np.rot90(self.img)
            self.coord[0,1], self.coord[1,1] = self.coord[1,1], self.coord[0,1] 

    def print(self):
        plt.imshow(self.img, cmap='Greys_r')
        plt.annotate(self.label, xy = (0, 0))
        plt.savefig("test.png" )
