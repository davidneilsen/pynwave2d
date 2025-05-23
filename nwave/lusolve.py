import numpy as np

class LinearSolveLU:
    def __init__(self, Ab):
        shp = np.shape(Ab)
        n = shp[1]
        N = n - 1
        self.n = n
        self.N = N

        PD = np.zeros(n)
        PC = np.zeros(n)
        PB = np.zeros(n)
        PA = np.zeros(n)

        PD[0:N-1] = Ab[0,2:]
        PC[0:N] = Ab[1,1:]
        PB[1:] = Ab[3,:-1]
        PA[2:] = Ab[4,:-2]

        UA = np.zeros(n)
        UB = np.zeros(n)
        UC = np.zeros(n)
        LA = np.zeros(n)
        LB = np.zeros(n)

        UA[0] = 1.0
        UB[0] = PC[0]
        UC[0] = PD[0]

        LB[1] = PB[1] / UA[0]
        UA[1] = 1 - UB[0]*LB[1]
        UB[1] = PC[1] - UC[0]*LB[1]
        UC[1] = PD[1]

        for i in range(2,N-1):
            LA[i] = PA[i] / UA[i-2]
            LB[i] = (PB[i] - UB[i-2]*LA[i]) / UA[i-1]
            UA[i] = 1 - UC[i-2]*LA[i] - UB[i-1]*LB[i]
            UB[i] = PC[i] - UC[i-1]*LB[i]
            UC[i] = PD[i]

        i = N-1
        LB[i] = (PB[i] - UB[i-2]*LA[i]) / UA[i-1]
        UA[i] = 1 - UC[i-2]*LA[i] - UB[i-1]*LB[i]
        UB[i] = PC[i] - UC[i-1]*LB[i]
        UC[i] = PD[i]

        i = N
        UA[i] = 1 - UC[i-2]*LA[i] - UB[i-1]*LB[i]
        UB[i] = PC[i] - UC[i-1]*LB[i]
        UC[i] = PD[i]

        self.PA = PA
        self.PB = PB
        self.PC = PC
        self.PD = PD

        self.UA = UA
        self.UB = UB
        self.UC = UC
        self.LA = LA
        self.LB = LB


    def solve(self, Z):
        n = self.n
        N = self.N

        UA = self.UA
        UB = self.UB
        UC = self.UC
        LA = self.LA
        LB = self.LB

        X = np.zeros(n)
        Y = np.zeros(n)

        Y[0] = Z[0]
        Y[1] = Z[1] - LB[1]*Y[0]
        for i in range(2, n):
            Y[i] = Z[i] - LA[i]*Z[i-2] - LB[i]*Z[i-1]

        X[N] = Y[N]/UA[N]
        X[N-1] = (Y[N-1] - UB[N-1]*X[N]) / UA[N-1]
        for i in range(N-2,0,-1):
            X[i] = (Y[i] - UB[i]*X[i+1] - UC[i]*X[i+2]) / UA[i]

        return X
