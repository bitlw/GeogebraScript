import os
import numpy as np
import sys
import argparse
import zipfile
import xml.dom.minidom
import shapely.geometry as sg
from shutil import copyfile


g_supportType = ['Segment', 'Vector', 'Polygon']
def ParseArgs():
    parser = argparse.ArgumentParser(description='GeoGebra script, convert graph from 3d calculator to 2d geometry')
    parser.add_argument('-d', '--dataLocation', help='location of the ggb file, eg: ./geogebraProject/test.ggb')

    args = parser.parse_args()

    if args.dataLocation is None:
        print('location of data file is required')
        exit
    else:
        if not os.path.exists(args.dataLocation):
            print(f'file {args.dataLocation} does not exist, exit')
        exit
    
    print(args)
    return args

def ReadPanes(root):
    panes = root.getElementsByTagName('pane')
    horizontal = 0.0
    vertical = 0.0
    #------------------------------------#
    # I can see different panes structure, and I also notice the structure of web version is different from app,
    # I'm now sure how I can accumulate panes, so just do a simple way based on what I can see from WEB version.
    #------------------------------------#

    # for pane in panes:
    #     orientation = pane.getAttribute('orientation')
    #     currentRate = float(pane.getAttribute('divider'))
    #     if orientation == '1':
    #         vertical += currentRate
    #     else:
    #         horizontal += currentRate

    vertical = float(panes[0].getAttribute('divider'))
    return horizontal, vertical

def ReadCoordSystem(coordSystem, name, defaultValue):
    scale = coordSystem.getAttribute(name)
    if scale == '':
        scale = defaultValue
    else:
        scale = float(scale)
    
    return scale

def CalculateAxisTransform(xZero, yZero, zZero, scale, yscale, zscale, xAngle, zAngle, yVerticalFlag = False):
    xAngle = xAngle / 180.0 * np.pi
    zAngle = zAngle / 180.0 * np.pi
    scale /= 50.0
    yscale /= 50.0
    zscale /= 50.0
    cosz = np.cos(zAngle)
    sinz = np.sin(zAngle)
    cosx = np.cos(xAngle)
    sinx = np.sin(xAngle)
    Rsw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tsw = np.array([xZero, yZero, zZero]).reshape(3, 1)

    localScale = np.array([[1, 0, 0], [0, yscale, 0], [0, 0, zscale]])
    Ry = np.eye(3)
    if yVerticalFlag:
        Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    dR = np.array([[1, 0, 0], [0, cosx, sinx], [0, -sinx, cosx]])
    dR = dR @ np.array([[cosz, sinz, 0], [-sinz, cosz, 0], [0, 0, 1]])
    
    R = scale * dR @ Ry @ Rsw
    t = R @ tsw
    R = R @ localScale
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3:4] = t

    return T

def CalculateCameraPose(xZero, yZero, zZero, scale, yscale, zscale, xAngle, zAngle, yVerticalFlag = False):
    Tcs = np.eye(4)
    Rcs = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    tcs = np.array([0, 0, 150]).reshape(3, 1)
    # Rcs = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    # tcs = np.array([0, 0, -500]).reshape(3, 1)
    Tcs[0:3, 0:3] = Rcs
    Tcs[0:3, 3:4] = tcs
    T = CalculateAxisTransform(xZero, yZero, zZero, scale, yscale, zscale, xAngle, zAngle, yVerticalFlag)

    return Tcs @ T

def ReadExpression(root):
    expressions = root.getElementsByTagName('expression')
    dictExpression = {}
    for expression in expressions:
        label = expression.getAttribute('label')
        exp = expression.getAttribute('exp')
        dictExpression[exp] = label
    return dictExpression

def ReadSortedAttribute(attributes, sort = True):
    labelList = []
    keys = attributes.keys()
    if sort:
        keys = sorted(attributes.keys())
    for key in keys:
        val = attributes[key].nodeValue
        labelList.append(val)
    return labelList


def ParseCommand(commands):
    dictCmdObject = {}
    dictCmdObject['Segment'] = {'selfObjectLabelList': [], 'pointLabelList': []}
    for cmd in commands:
        attribute = cmd.getAttribute('name')
        if attribute not in dictCmdObject:
            dictCmdObject[attribute] = {'selfObjectLabelList': [], 'pointLabelList': []}

        if attribute not in g_supportType:
            continue
        input = cmd.getElementsByTagName('input')[0]
        output = cmd.getElementsByTagName('output')[0]
        pointLabelList = ReadSortedAttribute(input.attributes)
        cmdLabelAll = ReadSortedAttribute(output.attributes)
        cmdLabel = cmdLabelAll[0]
        dictCmdObject[attribute]['selfObjectLabelList'].append(cmdLabel)
        dictCmdObject[attribute]['pointLabelList'].append(pointLabelList)

        if attribute == 'Polygon':
            cmdLabelAll = cmdLabelAll[1:]
            for i in range(len(cmdLabelAll)):
                i1 = i
                i2 = (i + 1) % len(cmdLabelAll)
                subLabel = cmdLabelAll[i]
                dictCmdObject['Segment']['selfObjectLabelList'].append(subLabel)
                dictCmdObject['Segment']['pointLabelList'].append([pointLabelList[i1], pointLabelList[i2]])
            pass

    return dictCmdObject

def BuildObjectFromCommand(dictCmdObject, dictPoints):
    for varName in dictPoints:
        coords = dictPoints[varName]['coords']
        cmd = f'{varName} = np.array([{coords[0]}, {coords[1]}, {coords[2]}])'
        exec(cmd)

    for objType in dictCmdObject:
        if objType not in g_supportType:
            continue

        objectList = dictCmdObject[objType]['pointLabelList']
        object3dList = []
        for obj in objectList:
            npObj = []
            for ptVar in obj:
                cmd = f'pt = {ptVar}'
                exec(cmd)
                exec('npObj.append(pt)')
            object3dList.append(npObj)

        dictCmdObject[objType]['object3dList'] = object3dList
    
    return dictCmdObject

def ProjectPoint(T, K, point3d):
    R = T[0:3, 0:3]
    t = T[0:3, 3:4]
    q = R @ point3d.reshape(3, 1) + t
    q /= q[2][0]
    uv = K @ q

    return uv[0][0], uv[1][0]

def ProjectObject(T, K, polygon3dList, isPolygon = True):
    sgPolygon2dList = []
    for polygon in polygon3dList:
        polygon2d = []
        for eachPt in polygon:
            u, v = ProjectPoint(T, K, eachPt)
            polygon2d.append([u, v])
        if isPolygon:
            sgPolygon2dList.append(sg.Polygon(polygon2d))
        else: # both segment and vector are line
            sgPolygon2dList.append(sg.LineString(polygon2d))

    return sgPolygon2dList

def MapLineStyleType(type):
    dictStyle = {'0': 0, '15': 1, '10': 2, '20': 3, '30': 4}
    return dictStyle[type]

def TagValid(tag):
    return tag != '' and len(tag) > 0

def ParseElements(root):
    elements = root.getElementsByTagName('element')
    dictPoints3d = {}
    dictElementInfo = {}
    for element in elements:
        elementType = element.getAttribute('type')
        varName = element.getAttribute('label')
        if varName not in dictElementInfo:
            show = element.getElementsByTagName('show')
            if TagValid(show) and show[0].getAttribute('object') == 'true':
                dictElementInfo[varName] = {}
                dictElementInfo[varName]['type'] = elementType
                objColor = element.getElementsByTagName('objColor')
                if TagValid(objColor):
                    color = ReadSortedAttribute(objColor[0].attributes, False)
                    dictElementInfo[varName]['objColor'] = [int(color[0]), int(color[1]), int(color[2])]
                
                pointSize = element.getElementsByTagName('pointSize')
                if TagValid(pointSize):
                    dictElementInfo[varName]['pointSize'] = int(pointSize[0].getAttribute('val'))

                lineStyle = element.getElementsByTagName('lineStyle')
                if TagValid(lineStyle):
                    dictElementInfo[varName]['thickness'] = int(lineStyle[0].getAttribute('thickness')) + 1
                    dictElementInfo[varName]['lineType'] = MapLineStyleType(lineStyle[0].getAttribute('type'))

        if elementType == 'point' or elementType == 'point3d':
            if varName in dictPoints3d:
                continue

            dictPoints3d[varName] = {}
            coords = element.getElementsByTagName('coords')[0]
            x = float(coords.getAttribute('x'))
            y = float(coords.getAttribute('y'))
            z = float(coords.getAttribute('z'))
            if elementType == 'point':
                z = 0
            w = coords.getAttribute('w')
            if w != '' and w != '0':
                w = float(w)
                x /= w
                y /= w
                z /= w
            pt = np.array([x, y, z])
            dictPoints3d[varName]['coords'] = pt

    return dictPoints3d, dictElementInfo

# np array
def IsPointOnLine3d(line3d, point3d):
    v1 = line3d[1] - line3d[0]
    v2 = point3d - line3d[0]
    return np.linalg.norm(np.cross(v1, v2)) < 1.0e-6

# np array
def Calculate3dIntersection(line3d, polygon3d):
    loopCount = min(3, len(polygon3d))
    assert loopCount >= 3
    for i in range(loopCount):
        polygonPoint = polygon3d[i]
        if IsPointOnLine3d(line3d, polygonPoint):
            return polygonPoint
    v1 = polygon3d[1] - polygon3d[0]
    v2 = polygon3d[2] - polygon3d[0]
    vVertical = np.cross(v1, v2)
    vp = line3d[1] - line3d[0]

    if np.abs(np.dot(vp, vVertical)) < 1.0e-6:
        return None # no intersection

    t = (np.dot(polygon3d[0], vVertical) - np.dot(line3d[0], vVertical)) / np.dot(vp, vVertical) 
    if t < 0 or t > 0.999: # give a tolerance for the line which is on polygon
        return None # no intersection
    else:
        return line3d[0] + t * (line3d[1] - line3d[0])
    
def SplitLines(sgLine2dList, sgPolygon2dList, npLine3dList, npPolygon3dList, T, K):
    lineIntersectionsList = []
    for i in range(len(sgLine2dList)):
        sgLine2d = sgLine2dList[i]
        # remove _coords to fit for different version of shapely 
        # intersectionsList = sgLine2d.coords._coords.tolist()
        intersectionsList = np.array(sgLine2d.coords).tolist()
        for j in range(len(sgPolygon2dList)):
            sgPolygon = sgPolygon2dList[j]
            if sgPolygon.intersects(sgLine2d):
                intersections = sgPolygon.intersection(sgLine2d)
                intersectionsList.extend(np.array(intersections.coords).tolist())

                intersection3d = Calculate3dIntersection(npLine3dList[i], npPolygon3dList[j])
                if intersection3d is not None:
                    u, v = ProjectPoint(T, K, intersection3d)
                    intersectionsList.extend([[u, v]])

        lineIntersectionsList.append(intersectionsList)
    return lineIntersectionsList

def Recover3dOnLine(uv, line3d, T, K):
    P1 = K @ (T[0:3, 0:3] @ line3d[0].reshape(3, 1) + T[0:3, 3:4])
    P2 = K @ (T[0:3, 0:3] @ line3d[1].reshape(3, 1) + T[0:3, 3:4])
    x1, y1, z1 = P1[0][0], P1[1][0], P1[2][0]
    x2, y2, z2 = P2[0][0], P2[1][0], P2[2][0]
    u, v = uv[0], uv[1]

    tmp_1 = x2 - x1 + u * (z1 - z2)
    tmp_2 = y2 - y1 + v * (z1 - z2)

    lambda_ = 1
    if np.abs(tmp_1) > np.abs(tmp_2):
        lambda_ = (u * z1 - x1) / tmp_1
    else:
        lambda_ = (v * z1 - y1) / tmp_2

    return line3d[0] + lambda_ * (line3d[1] - line3d[0])

def CheckVisible(point3d, point2d, sgPolygon2dList, npPolygon3dList, T):
    polygonCount = len(npPolygon3dList)
    R = T[0:3, 0:3]
    t = T[0:3, 3:4].reshape(3, 1)
    cameraCenter = - R.T @ t
    cameraCenter = cameraCenter.reshape(1, 3)[0]
    point = sg.Point(point2d[0], point2d[1])
    for i in range(polygonCount):
        polygon = sgPolygon2dList[i]
        if polygon.contains(point):
            if Calculate3dIntersection([cameraCenter, point3d], npPolygon3dList[i]) is not None:
                return False
    
    return True

def SortIntersection(intersection):
    # first element should be always at the first
    if len(intersection) <= 1:
        return intersection
    
    center = np.mean(intersection, axis=0)
    diff = intersection[0] - center
    if np.abs(diff[0]) > np.abs(diff[1]): # delta_x > delta_y
        if intersection[0][0] < center[0]: # p0 < center
            return sorted(intersection, key=lambda x:x[0])
        else:
            return sorted(intersection, key=lambda x:x[0], reverse=True)
    else:
        if intersection[0][1] < center[1]:
            return sorted(intersection, key=lambda x:x[1])
        else:
            return sorted(intersection, key=lambda x:x[1], reverse=True)

def DetectOcclusion(lineIntersectionsList, npLine3dList, sgPolygon2dList, npPolygon3dList, T, K):
    lineCount = len(lineIntersectionsList)
    segmentVisibleList = []
    updatedIntersectionList = []
    for i in range(lineCount):
        line = SortIntersection(lineIntersectionsList[i])
        segmentCount = len(line) - 1
        visibleList = []
        updatedIntersection = []
        for j in range(segmentCount):
            point2d_1 = np.array(line[j])
            point2d_2 = np.array(line[j + 1])
            distance = point2d_2 - point2d_1
            if np.linalg.norm(distance) < 0.001:
                continue

            updatedIntersection.append(line[j])
            center2d = (point2d_1 + point2d_2) / 2.0
            center3d = Recover3dOnLine(center2d, npLine3dList[i], T, K)
            visible = CheckVisible(center3d, center2d, sgPolygon2dList, npPolygon3dList, T)
            visibleList.append(visible)

        updatedIntersection.append(line[-1])
        updatedIntersectionList.append(updatedIntersection)
        segmentVisibleList.append(visibleList)

    return segmentVisibleList, updatedIntersectionList

def ExtractXML(ggbXMLFile):
    dom = xml.dom.minidom.parse(ggbXMLFile)
    root = dom.documentElement

    # read window size of rendering 
    horizontalRate, verticalRate = ReadPanes(root)
    window = root.getElementsByTagName('window')[0]
    width, height = int(window.getAttribute('width')), int(window.getAttribute('height'))
    renderWidth = width * (1 - verticalRate)
    renderHeight = height * (1 - horizontalRate)

    # read transformation of camera
    euclidianView3D = root.getElementsByTagName('euclidianView3D')[0]
    coordSystem = euclidianView3D.getElementsByTagName('coordSystem')[0]
    xZero = ReadCoordSystem(coordSystem, 'xZero', 0.0)
    yZero = ReadCoordSystem(coordSystem, 'yZero', 0.0)
    zZero = ReadCoordSystem(coordSystem, 'zZero', 0.0)
    scale = ReadCoordSystem(coordSystem, 'scale', 50.0)
    yscale = ReadCoordSystem(coordSystem, 'yscale', 50.0)
    zscale = ReadCoordSystem(coordSystem, 'zscale', 50.0)
    xAngle = ReadCoordSystem(coordSystem, 'xAngle', 0.0)
    zAngle = ReadCoordSystem(coordSystem, 'zAngle', 0.0)

    projectionType = euclidianView3D.getElementsByTagName('projection')[0]
    projectionType = projectionType.getAttribute('type')

    yAxisVertical = euclidianView3D.getElementsByTagName('yAxisVertical')
    yVerticalFlag = False
    if yAxisVertical != '' and len(yAxisVertical) > 0:
        yAxisVertical = yAxisVertical[0].getAttribute('val')
        yVerticalFlag = yAxisVertical == 'true'

    Tcw = CalculateCameraPose(xZero, yZero, zZero, scale, yscale, zscale, xAngle, zAngle, yVerticalFlag)
    f = 250
    # K = np.array([[f, 0, renderWidth / 2.0], [0, f, renderHeight / 2.0], [0, 0, 1]])
    K = np.array([[f, 0, 0], [0, -f, 0], [0, 0, 1]])
    
    # Read elements
    dictExpression = ReadExpression(root)
    dictPoints, dictElementInfo = ParseElements(root)
    dictCmdObject = ParseCommand(root.getElementsByTagName('command'))
    dictCmdObject = BuildObjectFromCommand(dictCmdObject, dictPoints)
    ProceedGeometry(dictCmdObject, Tcw, K)
    ProceedPoints(dictPoints, dictCmdObject, dictElementInfo, Tcw, K)
    Translate(dictCmdObject)
    commandList = BuildGeogebraCommand(dictCmdObject, dictElementInfo)
    
    return commandList

def ProceedPoints(dictPoints, dictCmdObject, dictElementInfo, T, K):
    point2dList = []
    selfObjectLabelList = []
    for key in dictPoints:
        if key not in dictElementInfo:
            continue
        u, v = ProjectPoint(T, K, dictPoints[key]['coords'])
        selfObjectLabelList.append(key)
        point2dList.append([u, v])
    
    dictCmdObject['Point'] = {'point2dList': point2dList, 'selfObjectLabelList': selfObjectLabelList}

def Translate(dictCmdObject):
    intersectionList = []
    for key in dictCmdObject:
        if key == 'Point':
            point2dList = dictCmdObject[key]['point2dList']
            intersectionList.extend(point2dList)
        elif key == 'Polygon':
            sgObject2dList = dictCmdObject[key]['sgObject2dList']
            for i in range(len(sgObject2dList)):
                coords = np.array(sgObject2dList[i].exterior.coords)
                intersectionList.extend(coords.tolist())
        elif 'intersectionList' in dictCmdObject[key]:
            for intersection in dictCmdObject[key]['intersectionList']:
                for eachPoint in intersection:
                    intersectionList.append(eachPoint)
        
    intersectionList = np.array(intersectionList)
    center = np.mean(intersectionList, axis=0)
    xymin = np.min(intersectionList, axis=0)
    xymax = np.max(intersectionList, axis=0)
    bboxLeftUp = np.array([xymin[0], xymax[1]])
    bboxRightDown = np.array([xymax[0], xymin[1]])
    dictCmdObject['Point']['point2dList'].append(bboxLeftUp.tolist())
    dictCmdObject['Point']['point2dList'].append(bboxRightDown.tolist())
    dictCmdObject['Point']['selfObjectLabelList'].append('Export_1')
    dictCmdObject['Point']['selfObjectLabelList'].append('Export_2')

    # move center = (0, 0)
    for key in dictCmdObject:
        if 'intersectionList' not in dictCmdObject[key] and key != 'Polygon' and key != 'Point':
            continue

        if key == 'Point':
            point2dList = dictCmdObject[key]['point2dList']
            for i in range(len(point2dList)):
                point2dList[i][0] -= center[0]
                point2dList[i][1] -= center[1]
        elif key == 'Polygon':
            sgNewPolygonList = []
            sgObject2dList = dictCmdObject[key]['sgObject2dList']
            # we have to create new polygons because we cannot update sg.Polygon directly
            for i in range(len(sgObject2dList)):
                coords = np.array(sgObject2dList[i].exterior.coords)
                coords -= center
                newPolygon = sg.Polygon(coords)
                sgNewPolygonList.append(newPolygon)
            dictCmdObject[key]['sgObject2dList'] = sgNewPolygonList
        else:
            for intersection in dictCmdObject[key]['intersectionList']:
                for eachPoint in intersection:
                    eachPoint[0] -= center[0]
                    eachPoint[1] -= center[1]


def ProceedGeometry(dictCmdObject, T, K):
    for key in dictCmdObject:
        if key not in g_supportType:
            continue

        dictCmdObject[key]['sgObject2dList'] = ProjectObject(T, K, dictCmdObject[key]['object3dList'], key == 'Polygon')
    
    doList = ['Segment', 'Vector']
    for key in doList:
        if key not in dictCmdObject:
            continue

        sgObject2dList = dictCmdObject[key]['sgObject2dList']
        object3dList = dictCmdObject[key]['object3dList']
        sgPolygon2dList = []
        polygon3dList = []

        if 'Polygon' in dictCmdObject:
            sgPolygon2dList = dictCmdObject['Polygon']['sgObject2dList']
            polygon3dList = dictCmdObject['Polygon']['object3dList']

        intersectionList = SplitLines(sgObject2dList, sgPolygon2dList, object3dList, polygon3dList, T, K)
        dictCmdObject[key]['visibleList'], dictCmdObject[key]['intersectionList'] = DetectOcclusion(intersectionList, object3dList, sgPolygon2dList, polygon3dList, T, K)


def BuildGeogebraCommand(dictCmdObject, dictElementInfo):
    commandList = []
    vecIndex = 0
    for key in dictCmdObject:
        labelList = dictCmdObject[key]['selfObjectLabelList']
        if len(labelList) <= 0:
            continue

        if key == 'Point':
            point2dList = dictCmdObject[key]['point2dList']
            for i in range(len(point2dList)):
                labelName = 'AP' + str(vecIndex)
                if labelList[i] in dictElementInfo:
                    CreateObject(labelName, '', [point2dList[i]], commandList)
                    objInfo = dictElementInfo[labelList[i]]
                    SetColor(objInfo, labelName, commandList)
                    SetAttribute(objInfo, 'SetPointSize', labelName, 'pointSize', commandList)
                else: # for Export_1 and Export_2
                    labelName = labelList[i]
                    CreateObject(labelName, '', [point2dList[i]], commandList)
                    cmd = f'SetConditionToShowObject({labelName}, False)'
                    commandList.append(cmd)
                
                cmd = f'ShowLabel({labelName}, False)'
                commandList.append(cmd)
                vecIndex += 1
        elif key == 'Polygon':
            sgObject2dList = dictCmdObject[key]['sgObject2dList']
            for i in range(len(sgObject2dList)):

                if labelList[i] not in dictElementInfo: # means invisible
                    continue

                objInfo = dictElementInfo[labelList[i]]
                sgPolygon = sgObject2dList[i]
                coords = np.array(sgPolygon.exterior.coords)
                labelName = 'AP' + str(vecIndex)
                CreateObject(labelName, key, coords, commandList)
                SetLineAttribute(labelName, objInfo, True, commandList)
                vecIndex += 1
        else:
            visibleList = dictCmdObject[key]['visibleList']
            intersectionList = dictCmdObject[key]['intersectionList']

            for i in range(len(intersectionList)):
                intersections = intersectionList[i]

                for j in range(len(visibleList[i])):
                    if labelList[i] not in dictElementInfo: # means invisible
                        continue

                    objInfo = dictElementInfo[labelList[i]]

                    labelName = 'A' + str(vecIndex)
                    objTypeInCmd = 'Segment'
                    if key == 'Vector' and j == len(visibleList[i]) - 1:
                        objTypeInCmd = 'Vector'

                    point2dList = [intersections[j], intersections[j + 1]]
                    CreateObject(labelName, objTypeInCmd, point2dList, commandList)
                    SetLineAttribute(labelName, objInfo, visibleList[i][j], commandList)
                    vecIndex += 1

    commandList = sorted(commandList)
    return commandList    

def CreateObject(labelName, objTypeInCmd, point2dList, commandList):
    cmd = f'{labelName} = {objTypeInCmd}'
    cmd += '('
    for point2d in point2dList:
        point2d = np.round(point2d, 2)
        cmd += f'({str(point2d[0])}, {str(point2d[1])}), '

    cmd = cmd[0:-2]
    cmd += ')'
    commandList.append(cmd)

def SetLineAttribute(labelName, objInfo, visibleFlag, commandList):
    # set attribute
    SetAttribute(objInfo, 'SetLineThickness', labelName, 'thickness', commandList)
    SetColor(objInfo, labelName, commandList)

    if visibleFlag:
        SetAttribute(objInfo, 'SetLineStyle', labelName, 'lineType', commandList)
    else:
        SetAttribute(objInfo, 'SetLineStyle', labelName, 'lineType', commandList, 1)


def SetColor(objInfo, labelName, commandList):
    if 'objColor' in objInfo:
        value = objInfo['objColor']
        r, g, b = str(value[0] / 255), str(value[1] / 255), str(value[2] / 255)
        cmd = f'SetColor({labelName}, {r}, {g}, {b})'
        commandList.append(cmd)

def SetAttribute(objInfo, geobebraCmd, labelName, attributeName, commandList, valueInput = None):
    value = None
    if attributeName in objInfo:
        value = str(objInfo[attributeName])

    if valueInput is not None:
        value = valueInput

    if value is not None:
        cmd = f'{geobebraCmd}({labelName}, {value})'
        commandList.append(cmd)

def UnzipAndExtractXML(ggbFile):
    inputFolder, baseName = os.path.split(ggbFile)
    fileName = os.path.splitext(baseName)[0]
    rstFolder = os.path.join(inputFolder, fileName)
    dstFile = os.path.join(rstFolder, fileName + '.zip')
    os.makedirs(rstFolder, exist_ok=True)
    if not os.path.exists(dstFile):
        copyfile(ggbFile, dstFile)

    ggbXMLFile = os.path.join(rstFolder, 'geogebra.xml')
    if not os.path.exists(ggbXMLFile):
        with zipfile.ZipFile(dstFile, 'r') as zip_ref:
            zip_ref.extractall(rstFolder)

    commandList = ExtractXML(ggbXMLFile)
    commandList = np.array(commandList)
    np.savetxt(f'{rstFolder}/commandList.txt', commandList, fmt="%s")

if __name__ == "__main__":
    args = ParseArgs()
    UnzipAndExtractXML(args.dataLocation)
    