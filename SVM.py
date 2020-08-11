import os
import numpy as np
import cv2 as cv

# SVM does is to find a straight line (or hyperplane) with largest minimum distance to the training samples
# Linear separable data can be classified with decision boundary(hyperplane)
# Most close two(multiple) data from opposite classes are "Support Vectors"
# The perfect decision boundary passing through support vectors called "Support Planes"
# Distance from each close data to support plane = 1/ ||w||
# Margin width = 2**Distance = 2 / ||w||
# SVM dose is to maximize distance 1 / ||w||
# Non-Linear separable data --> lift dimension(kernel) to find hyperplane

# HOG feature vectors: Histogram of Oriented Gradient
# 1. Sobel first order derivation along x and y to get gradient magnitude along x and y
# 2. cv.cartToPolar to convert gradient into gradient magnitude and direction
# 3. quantize gradient direction to 16 integer values(bins) each of four sub-region of images
# 4. total image HOG feature vector contains 16*4 = 64 values

kernels = {
    -1: 'CUSTOM',
    0: 'LINEAR',
    1: 'POLY',
    2: 'RBF',
    3: 'SIGMOID',
    4: 'CHI2',
    5: 'INTER'
}

types = {
  100: 'C_SVC',
  101: 'NU_SVC',
  102: 'ONE_CLASS',
  103: 'EPS_SVR',
  104: 'NU_SVR'
}


class SVMOCR(object):

    svm = cv.ml.SVM_create()
    __affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

    def __new__(cls, *args, **kwargs):
        cls.setSVM()
        return super(SVMOCR, cls).__new__(cls)

    def __init__(self, imgPath, affineWidth=20, HOGBins=16):
        self.img = cv.imread(imgPath)
        if self.img is None:
            raise Exception(f"Can't read image from given path: {os.path.abspath(imgPath)}")
        self.img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.afw = affineWidth
        self.bins = HOGBins
        print(f'Affine width: {self.afw}', f'HOG bins: {self.bins}', sep='\n')

    @classmethod
    def setSVM(cls, Kernel=cv.ml.SVM_LINEAR, Type=cv.ml.SVM_C_SVC, C=2.67, Gamma=5.383, Nu=0.5, P=1):
        cls.svm.setKernel(Kernel)
        cls.svm.setType(Type)
        cls.svm.setC(C)
        cls.svm.setGamma(Gamma)
        cls.svm.setNu(Nu)
        cls.svm.setP(P)
        print(f'Kernel: {kernels[Kernel]}',
              f'Type: {types[Type]}',
              f'C: {C}',
              f'Gamma: {Gamma}',
              f'Nu: {Nu}',
              f'P: {P}',
              sep='\n')

    def ImgSplit(self, Xcount=100, Ycount=50):
        """
        . Split 'digits.png' data into digit cells
        . First half of the cells are for training, and the rest are for testing
        """
        train_cells = [cell for col in np.hsplit(self.img_gray, Xcount)[:int(0.5*Xcount)] for cell in np.vsplit(col, Ycount)]
        test_cells = [cell for col in np.hsplit(self.img_gray, Xcount)[int(0.5*Xcount):] for cell in np.vsplit(col, Ycount)]
        return train_cells, test_cells

    def DataProcess(self, data_cells, response=False):
        """
        . Deskew data and Get data HOG feature vectors
        :param data_cells: list of data image arrays
        :param response: labels
        :return: deskewed data HOG feature vectors, or (HOG features, Responses) if response = True
        """
        deskewed = [self.DeSkew(cell) for cell in data_cells]
        hogdata = [self.HOG(cell) for cell in deskewed]
        hogdata = np.float32(hogdata)
        if response:
            responses = np.repeat(np.arange(10), 5)[:, np.newaxis]
            responses = np.vstack(np.broadcast_arrays((responses,)*50)[0])
            return hogdata, responses
        else:
            return hogdata

    def DeSkew(self, img):
        """
        . Deskew hand-written character image
        """
        mom = cv.moments(img)
        if abs(mom['mu02']) < 0.01:
            return img.copy()
        skew = mom['mu11'] / mom['mu02']
        affine_M = np.array([
            [1, skew, -0.5*self.afw*skew],
            [0, 1, 0]
        ], dtype=np.float32)
        img = cv.warpAffine(img, affine_M, (self.afw, self.afw), flags=self.__affine_flags)
        return img

    def HOG(self, img):
        """
        . Get HOG feature vectors
        """
        # get x and y directional gradients using Sobel filter
        gx, gy = cv.Sobel(img, cv.CV_32F, 1, 0), cv.Sobel(img, cv.CV_32F, 0, 1)
        # convert gx, gy to polar coordinate
        mag, ang = cv.cartToPolar(gx, gy)
        # quantize bin values of vector directions(angles) to specified bins
        ang_bin = np.int32(self.bins*ang/(2*np.pi))
        size = int(0.5*self.afw)
        sub_regions = [
            (ang_bin[:size, :size], mag[:size, :size]),
            (ang_bin[size:, :size], mag[size:, :size]),
            (ang_bin[:size, size:], mag[:size, size:]),
            (ang_bin[size:, size:], mag[size:, size:])
        ]
        hists = [np.bincount(a.ravel(), m.ravel(), self.bins) for a, m in sub_regions]
        hist = np.hstack(hists)
        return hist


if __name__ == '__main__':
    OCR = SVMOCR('./sample/digits.png',HOGBins=16)
    OCR.setSVM(Type=101, Nu=0.15)


    trainCells, testCells = OCR.ImgSplit(Xcount=100, Ycount=50)
    trainData, responses = OCR.DataProcess(trainCells, response=True)
    testData = OCR.DataProcess(testCells)

    OCR.svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
    OCR.svm.save('svm_data.dat')

    ret, prediction = OCR.svm.predict(testData)
    accuracy = np.count_nonzero(prediction == responses) / prediction.size
    print(f'Accuracy: {100 * accuracy}%')

    img = OCR.img.copy()
    for i, p in enumerate(prediction):
        p_val = p[0]
        t_val = responses[i]
        correct = p_val == t_val
        color = (0, 255, 0) if correct else (0, 0, 255)
        origin_x = 1000+20*(i%50) + 1
        origin_y = 0+20*(i//50) + 1
        cv.rectangle(img,(origin_x, origin_y), (origin_x+18, origin_y+18), color, 2)

    cv.imshow(f'result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


