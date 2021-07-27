class WayPoints:
    def __init__(self, textFile):
        self.wayPoints = []
        self.textPath = textFile
        self.refresh()

    def refresh(self):
        f = open(self.textPath, "r")
        lines = (f.read()).split('\n')
        for line in lines:
            coords = line.split(',')
            self.wayPoints.append(Point(float(coords[0]), float(coords[1])))

    def returnWayPoints(self):
        return self.wayPoints


class Point:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.hit = False
