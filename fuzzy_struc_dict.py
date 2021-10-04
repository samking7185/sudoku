import numpy as np

from fuzzy_dict import Membership, MembershipArray, Rulebase, Defuzz


class FIS:
    def __init__(self, params):
        """
        Class to create a 2 input 1 output fuzzy inference system
        Calls fuzzy tool from fuzzy.py

        """
        self.params = params
        in1MF = []
        in2MF = []
        out3MF = []

        for key in self.params["MF1"]:
            self.params["MF1"][key].update({"object": Membership(self.params["MF1"][key]["value"])})

        for key in self.params["MF2"]:
            self.params["MF2"][key].update({"object": Membership(self.params["MF2"][key]["value"])})

        for key in self.params["OUT"]:
            self.params["OUT"][key].update({"object": MembershipArray(self.params["OUT"][key]["value"],
                                            np.linspace(params["BOUNDS"][0], params["BOUNDS"][1], num=100))})

        self.rulebase = Rulebase(self.params["RULES"])

    def compute(self, in1: float, in2: float):
        params = self.params
        Fr = self.rulebase

        input1 = []
        input2 = []

        for key in params["MF1"]:
            if params["MF1"][key]["shape"] == "gauss":
                mu = params["MF1"][key]["object"].gauss(in1)
                params["MF1"][key].update({"mu": mu})
            elif params["MF1"][key]["shape"] == "triangle":
                mu = params["MF1"][key]["object"].triangle(in1)
                params["MF1"][key].update({"mu": mu})

            elif params["MF1"][key]["shape"] == "trapezoid":
                mu = params["MF1"][key]["object"].trapezoid(in1)
                params["MF1"][key].update({"mu": mu})

        for key in params["MF2"]:
            if params["MF2"][key]["shape"] == "gauss":
                mu = params["MF2"][key]["object"].gauss(in2)
                params["MF2"][key].update({"mu": mu})
            elif params["MF2"][key]["shape"] == "triangle":
                mu = params["MF2"][key]["object"].triangle(in2)
                params["MF2"][key].update({"mu": mu})

            elif params["MF2"][key]["shape"] == "trapezoid":
                mu = params["MF2"][key]["object"].trapezoid(in2)
                params["MF2"][key].update({"mu": mu})

        rule_combos = []
        for key in params["RULES"]:
            mf1name = params["RULES"][key]["mf"][0]
            mf2name = params["RULES"][key]["mf"][1]
            mf3name = params["RULES"][key]["mf"][2]

            mf1 = params["MF1"][mf1name]["mu"]
            mf2 = params["MF2"][mf2name]["mu"]
            rule_temp = 0

            if params["RULES"][key]["oper"] == "OR":
                rule_temp = Fr.OR_rule([mf1, mf2])
            elif params["RULES"][key]["oper"] == "AND":
                rule_temp = Fr.AND_rule([mf1, mf2])

            if "rule_list" not in params["OUT"][mf3name]:
                params["OUT"][mf3name].update({"rule_list": []})

            params["OUT"][mf3name]["rule_list"].append(rule_temp)

        for key in params["OUT"]:
            temp_out = Fr.OR_rule(params["OUT"][key]["rule_list"])
            params["OUT"][key].update({"output": temp_out})

        output = Defuzz(params["OUT"], params["BOUNDS"])
        return output.out
