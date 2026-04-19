class Airfoil:
    def __init__(self, AR, alpha):
        self.AR = AR
        self.CL = [None] * len(alpha)
        self.CDi = [None] * len(alpha)
        self.CD = [None] * len(alpha)