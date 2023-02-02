import GeoGebra3dTo2D
import numpy as np

def VerifyAxisTransform(angles, scaleList, position, yVerticalFlag, Pw, Q):
    xAngle, zAngle = angles[0], angles[1]
    scale, yScale, zScale = scaleList[0], scaleList[1], scaleList[2]
    xZero, yZero, zZero = position[0], position[1], position[2]
    T = GeoGebra3dTo2D.CalculateAxisTransform(xZero, yZero, zZero, scale, yScale, zScale, xAngle, zAngle, yVerticalFlag)
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]
    Pw = np.array(Pw).reshape(3, 1)
    Q = np.array(Q).reshape(3, 1)
    Ps = R @ Pw + t
    delta = Ps - Q
    err = delta.T @ delta
    err = err[0][0]
    assert err < 1.0e-4

def test_CalculateAxisTransform():
    VerifyAxisTransform([90, 90], [50, 50, 50], [2, 1, -1.5], 0, [1, 2, 0], [3, -1.5, -3])
    VerifyAxisTransform([90, 90], [50, 50, 50], [2, 1, -1.5], 1, [1, 2, 0], [3, 3, -1.5])
    VerifyAxisTransform([90, 90], [50, 50, 50], [2, 1, -1.5], 0, [1, 2, 3], [3, 1.5, -3])
    VerifyAxisTransform([90, 90], [50, 50, 50], [2, 1, -1.5], 1, [1, 2, 3], [3, 3, 1.5])

def test_CalculateAxisTransformScale():
    VerifyAxisTransform([90, 90], [100, 50, 50], [0, 0, -1.5], 0, [1, 2, 3], [2, 3, -4])
    VerifyAxisTransform([90, 90], [100, 50, 50], [0, 0, -1.5], 1, [1, 2, 3], [2, 4, 3])

def test_PointOnLine():
    assert GeoGebra3dTo2D.IsPointOnLine3d(np.array([[0, 0, 0], [1, 1, 1]]), np.array([2, 2, 2])) == True
    assert GeoGebra3dTo2D.IsPointOnLine3d(np.array([[0, 0, 0], [1, 1, 1]]), np.array([0.2, 0.2, 0.2])) == True
    assert GeoGebra3dTo2D.IsPointOnLine3d(np.array([[0, 0, 0], [1, 1, 1]]), np.array([2, 2, 2.5])) == False

def VerifyIntersection(line3d, polygon3d, expectIntersection):
    intersection = GeoGebra3dTo2D.Calculate3dIntersection(line3d, polygon3d)
    if intersection is None:
        assert expectIntersection is None
    else:
        assert np.linalg.norm(intersection - expectIntersection) < 1.0e-3

def test_3dIntersection():
    line3d = np.array([[0, 0, 0], [2, 0, 0]])
    polygon3d = np.array([[1, 1, 1], [1, 2, 0], [1, -0.9, 0.5]])
    expect = np.array([1, 0, 0])
    VerifyIntersection(line3d, polygon3d, expect)

def test_3dIntersection2():
    line3d = np.array([[-4, -6, 4], [0, 2.24, 0]])
    polygon3d = np.array([[0, 0, 4], [0, -5, 0], [-6, 0, 0]])
    expect = np.array([-1.747, -1.36, 1.747])
    VerifyIntersection(line3d, polygon3d, expect)

def test_3dIntersectionNone():
    line3d = np.array([[0, 0, 0], [0.5, 0, 0]])
    polygon3d = np.array([[1, 1, 1], [1, 2, 0], [1, -0.9, 0.5]])
    expect = None
    VerifyIntersection(line3d, polygon3d, expect)
