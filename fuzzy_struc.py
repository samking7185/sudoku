import numpy as np

from fuzzy import Membership, MembershipArray, Rulebase, Defuzz


class FIS:
    def __init__(self, params):
        """
        Class to create a 2 input 1 output fuzzy inference system
        Calls fuzzy tool from fuzzy.py

        User instantiates the class with "params" which is a dictionary

        params = {
        "MF1": [(0, 0, 2), (1, 2, 4, 4)],
        "MF2": [(0, 0, 2), (1, 2, 4, 4)],
        "OUT": [(0, 0, 1), (0, 1, 1)],
        "RULES": [0, 1, 0, 1],
        "BOUNDS": [0, 1]
        }

        The membership functions take the form of tuples (x1, x2, x3, x4)
        2 values uses a gaussian membership function
        3 values uses a triangular membership function
        4 values uses a trapezoidal membership function

        The rules take the form of integers, rules must be specified for all combination of inputs
        The integer delineating which output membership function corresponds to the combination of inputs
        There needs to be nMF1 x nMF2 rules

        For 2 inputs membership functions:
        The outputs are evaluated as [MF1,1 and MF2,1]  [MF1,1 and MF2,2]  [MF1,2 and MF2,1]  [MF1,2 and MF2,2]

        For 3 inputs membership functions:
        The outputs are evaluated as [MF1,1 and MF2,1]  [MF1,1 and MF2,2]  [MF1,1 and MF2,3]  etc.
        """
        self.params = params
        in1MF = []
        in2MF = []
        out3MF = []

        for mf1 in params["MF1"]:
            in1MF_temp = Membership(mf1)
            in1MF.append(in1MF_temp)

        for mf2 in params["MF2"]:
            in2MF_temp = Membership(mf2)
            in2MF.append(in2MF_temp)

        for mf3 in params["OUT"]:
            out3MF_temp = MembershipArray(mf3, np.linspace(params["BOUNDS"][0], params["BOUNDS"][1], num=100))
            out3MF.append(out3MF_temp)

        self.params.update({"MF1obj": in1MF, "MF2obj": in2MF, "MF3obj": out3MF})
        self.rulebase = Rulebase()

    def compute(self, in1: float, in2: float):
        params = self.params
        Fr = self.rulebase

        input1 = []
        input2 = []

        for mf in params["MF1obj"]:
            if len(mf.params) == 2:
                input1.append(mf.gauss(in1))
            elif len(mf.params) == 3:
                input1.append(mf.triangle(in1))
            else:
                input1.append(mf.trapezoid(in1))

        for mf in params["MF2obj"]:
            if len(mf.params) == 2:
                input2.append(mf.gauss(in2))
            elif len(mf.params) == 3:
                input2.append(mf.triangle(in2))
            else:
                input2.append(mf.trapezoid(in2))

        rule_combos = []
        for r1 in input1:
            for r2 in input2:
                rule_temp = Fr.AND_rule([r1, r2])
                rule_combos.append(rule_temp)

        full_out_array = [[], [], [], [], [], [], []]

        for rule, combo in zip(params["RULES"], rule_combos):
            full_out_array[rule].append(combo)

        out_array = [out for out in full_out_array if out != []]

        mu_array = list(map(Fr.OR_rule, out_array))

        output = Defuzz(mu_array, params["MF3obj"], params["BOUNDS"])
        return output.out
