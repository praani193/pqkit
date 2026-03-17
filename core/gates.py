import numpy as np


# ==========================================================
# PARAMETER CLASS (For Training)
# ==========================================================

class Parameter:
    def __init__(self, value):
        self.value = value


# ==========================================================
# BASE GATE
# ==========================================================

class Gate:
    def __init__(self, name="GenericGate"):
        self.name = name

    def apply(self, statevector):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name} Gate"


# ==========================================================
# SINGLE QUBIT GATE
# ==========================================================

class SingleQubitGate(Gate):
    def __init__(self, matrix, target, name="SingleQubitGate"):
        super().__init__(name)
        self.matrix = matrix
        self.target = target

    def apply(self, sv):
        state = sv.state
        dim = sv.dim
        mask = 1 << self.target

        for i in range(dim):
            if (i & mask) == 0:
                j = i | mask

                a = state[i]
                b = state[j]

                state[i] = self.matrix[0, 0]*a + self.matrix[0, 1]*b
                state[j] = self.matrix[1, 0]*a + self.matrix[1, 1]*b


# ==========================================================
# PARAMETERIZED ROTATION GATE
# ==========================================================

class ParameterizedRY(Gate):
    def __init__(self, param, target):
        super().__init__("ParamRY")
        self.param = param
        self.target = target

    def apply(self, sv):
        theta = self.param.value

        c = np.cos(theta/2)
        s = np.sin(theta/2)

        state = sv.state
        dim = sv.dim
        mask = 1 << self.target

        for i in range(dim):
            if (i & mask) == 0:
                j = i | mask

                a = state[i]
                b = state[j]

                state[i] = c*a - s*b
                state[j] = s*a + c*b


# ==========================================================
# MULTI QUBIT GATE
# ==========================================================

class MultiQubitGate(Gate):
    def __init__(self, name, qubits):
        super().__init__(name)
        self.qubits = qubits

    def apply(self, sv):
        if sv.state.ndim == 2:
            for b in range(sv.batch_size):
                self._apply_single(sv.state[b], sv.dim)
            return
        self._validate(sv.n_qubits)

        if self.name == "CNOT":
            self._apply_cnot(sv)
        elif self.name == "CZ":
            self._apply_cz(sv)
        elif self.name == "SWAP":
            self._apply_swap(sv)
        elif self.name == "TOFFOLI":
            self._apply_toffoli(sv)
        elif self.name == "CRY":
            self._apply_cry(sv)
        elif self.name == "CRX":
            self._apply_crx(sv)
        elif self.name == "CRZ":
            self._apply_crz(sv)
        else:
            raise ValueError(f"Unknown multi-qubit gate: {self.name}")

    def _validate(self, total):
        for q in self.qubits:
            if isinstance(q, int):
                if q < 0 or q >= total:
                    raise ValueError("Qubit index out of range")

    # ---------- CNOT ----------
    def _apply_cnot(self, sv):
        state = sv.state
        dim = sv.dim
        c, t = self.qubits

        cm = 1 << c
        tm = 1 << t

        for i in range(dim):
            if (i & cm):
                j = i ^ tm
                if i < j:
                    state[i], state[j] = state[j], state[i]

    # ---------- CZ ----------
    def _apply_cz(self, sv):
        state = sv.state
        dim = sv.dim
        c, t = self.qubits

        cm = 1 << c
        tm = 1 << t

        for i in range(dim):
            if (i & cm) and (i & tm):
                state[i] *= -1

    # ---------- SWAP ----------
    def _apply_swap(self, sv):
        state = sv.state
        dim = sv.dim
        q1, q2 = self.qubits

        m1 = 1 << q1
        m2 = 1 << q2

        for i in range(dim):
            if bool(i & m1) != bool(i & m2):
                j = i ^ (m1 | m2)
                if i < j:
                    state[i], state[j] = state[j], state[i]

    # ---------- TOFFOLI ----------
    def _apply_toffoli(self, sv):
        state = sv.state
        dim = sv.dim
        c1, c2, t = self.qubits

        m1 = 1 << c1
        m2 = 1 << c2
        mt = 1 << t

        for i in range(dim):
            if (i & m1) and (i & m2):
                j = i ^ mt
                if i < j:
                    state[i], state[j] = state[j], state[i]

    # ---------- CRY ----------
    def _apply_cry(self, sv):
        state = sv.state
        dim = sv.dim
        c, t, theta = self.qubits

        cm = 1 << c
        tm = 1 << t

        cos = np.cos(theta/2)
        sin = np.sin(theta/2)

        for i in range(dim):
            if (i & cm) and (i & tm) == 0:
                j = i | tm

                a = state[i]
                b = state[j]

                state[i] = cos*a - sin*b
                state[j] = sin*a + cos*b

    # ---------- CRX ----------
    def _apply_crx(self, sv):
        state = sv.state
        dim = sv.dim
        c, t, theta = self.qubits

        cm = 1 << c
        tm = 1 << t

        cos = np.cos(theta/2)
        sin = -1j*np.sin(theta/2)

        for i in range(dim):
            if (i & cm) and (i & tm) == 0:
                j = i | tm

                a = state[i]
                b = state[j]

                state[i] = cos*a + sin*b
                state[j] = sin*a + cos*b

    # ---------- CRZ ----------
    def _apply_crz(self, sv):
        state = sv.state
        dim = sv.dim
        c, t, theta = self.qubits

        cm = 1 << c
        tm = 1 << t

        p0 = np.exp(-1j*theta/2)
        p1 = np.exp(1j*theta/2)

        for i in range(dim):
            if (i & cm):
                if (i & tm):
                    state[i] *= p1
                else:
                    state[i] *= p0


# ==========================================================
# FACTORY FUNCTIONS
# ==========================================================

def X(t): return SingleQubitGate(np.array([[0,1],[1,0]],complex), t, "X")
def Y(t): return SingleQubitGate(np.array([[0,-1j],[1j,0]],complex), t, "Y")
def Z(t): return SingleQubitGate(np.array([[1,0],[0,-1]],complex), t, "Z")
def H(t): return SingleQubitGate((1/np.sqrt(2))*np.array([[1,1],[1,-1]],complex), t, "H")

def RX(theta,t):
    m=np.array([[np.cos(theta/2),-1j*np.sin(theta/2)],
                [-1j*np.sin(theta/2),np.cos(theta/2)]],complex)
    return SingleQubitGate(m,t,"RX")

def RY(theta,t):
    m=np.array([[np.cos(theta/2),-np.sin(theta/2)],
                [np.sin(theta/2),np.cos(theta/2)]],complex)
    return SingleQubitGate(m,t,"RY")

def RZ(theta,t):
    m=np.array([[np.exp(-1j*theta/2),0],
                [0,np.exp(1j*theta/2)]],complex)
    return SingleQubitGate(m,t,"RZ")

def CNOT(c,t): return MultiQubitGate("CNOT",[c,t])
def CZ(c,t): return MultiQubitGate("CZ",[c,t])
def SWAP(q1,q2): return MultiQubitGate("SWAP",[q1,q2])
def TOFFOLI(c1,c2,t): return MultiQubitGate("TOFFOLI",[c1,c2,t])

def CRY(c,t,theta): return MultiQubitGate("CRY",[c,t,theta])
def CRX(c,t,theta): return MultiQubitGate("CRX",[c,t,theta])
def CRZ(c,t,theta): return MultiQubitGate("CRZ",[c,t,theta])