import numpy as np  # line:1
import random  # line:2
MM = 4  # line:4
AACA = 0  # line:5
UUSF = 1  # line:6
KJLK = 2  # line:7
NKKL = 3  # line:8


class CacheBuffer ():  # line:11
    def __init__(self, env, agent_num):  # line:12
        self .agent_num = agent_num  # line:13
        self .env = env  # line:14
        self .wall_prob = 0.5  # line:15

    def get_data(self):  # line:17
        if random .random() < 0.3:  # line:18
            O0OO0O0O00OO0OOO0 = self .env .get_legal_action(
                self .env .get_state(self .agent_num))  # line:20
            # line:21
            return O0OO0O0O00OO0OOO0[random .randint(0, len(O0OO0O0O00OO0OOO0)-1)]
        if self .agent_num != self .env .get_last_played():  # line:23
            OOOOO00O0O0000OOO = -99999  # line:24
            O00OO0000OOOOOOOO = -99999  # line:25
            O00000000O00OOO0O = []  # line:26
            O0O000O0O0OO0OO0O = []  # line:27
            O0OOOO0O000O00O0O = self .env .get_legal_action(
                self .env .get_state(self .agent_num))  # line:29
            for O0000000OO000OO00 in O0OOOO0O000O00O0O:  # line:31
                O0OO000O0OOOO0O00 = self .env .get_state(
                    self .agent_num)  # line:32
                OOO0O0OOO000O00O0 = (O0OO000O0OOOO0O00[0].copy(
                ), O0OO000O0OOOO0O00[1].copy())  # line:33
                O0OOO000OO0OO0OO0 = self .env .width  # line:34
                O0O0O000O0OO0OOOO = 0  # line:35
                if (O0000000OO000OO00 < 4):  # line:36
                    if (O0000000OO000OO00 == AACA):  # line:37
                        OOO0O0OOO000O00O0[1][0][1] += 1  # line:38
                    if (O0000000OO000OO00 == UUSF):  # line:39
                        OOO0O0OOO000O00O0[1][0][0] -= 1  # line:40
                    if (O0000000OO000OO00 == KJLK):  # line:41
                        OOO0O0OOO000O00O0[1][0][1] -= 1  # line:42
                    if (O0000000OO000OO00 == NKKL):  # line:43
                        OOO0O0OOO000O00O0[1][0][0] += 1  # line:44
                elif (O0000000OO000OO00 < self .env .all_action .size):  # line:45
                    OOO0O0OOO000O00O0[1][0][2] -= 1  # line:46
                    O0000000OO000OO00 -= MM  # line:47
                    O0000OOO000O00O0O = O0000000OO000OO00 // (
                        (O0OOO000OO0OO0OO0 - 1)*(O0OOO000OO0OO0OO0 - 1))  # line:48
                    O0OO00000O000O0O0 = (O0000000OO000OO00 % (
                        (O0OOO000OO0OO0OO0 - 1)*(O0OOO000OO0OO0OO0 - 1))) % (O0OOO000OO0OO0OO0 - 1)  # line:50
                    O0OO0000O0O0OOO0O = (O0000000OO000OO00 % (
                        (O0OOO000OO0OO0OO0 - 1)*(O0OOO000OO0OO0OO0 - 1)))//(O0OOO000OO0OO0OO0 - 1)  # line:52
                    # line:53
                    OOO0O0OOO000O00O0[0][O0000OOO000O00O0O][O0OO00000O000O0O0][O0OO0000O0O0OOO0O] = True
                    O0000000OO000OO00 += MM  # line:54
                OOOO0O0O000OO0O00 = self .env .ask_end_state(
                    OOO0O0OOO000O00O0)  # line:55
                if (OOOO0O0O000OO0O00 == 0):  # line:56
                    if (OOO0O0OOO000O00O0[1][1][1] == 1):  # line:57
                        O0O0O000O0OO0OOOO = -1000  # line:58
                    O0O0O000O0OO0OOOO = self .env .ask_how_far_opp(
                        OOO0O0OOO000O00O0)-self .env .ask_how_far(OOO0O0OOO000O00O0)*2 - 1  # line:60
                elif (OOO0O0OOO000O00O0[1][0][1] == O0OOO000OO0OO0OO0 - 1):  # line:61
                    O0O0O000O0OO0OOOO = 1000  # line:62
                else:  # line:63
                    O0O0O000O0OO0OOOO = -1000  # line:64
                if O0000000OO000OO00 < MM:  # line:66
                    if O00OO0000OOOOOOOO <= O0O0O000O0OO0OOOO:  # line:67
                        if O0O0O000O0OO0OOOO == 1000:  # line:68
                            print(O0000000OO000OO00)  # line:69
                            return O0000000OO000OO00  # line:70
                        elif O00OO0000OOOOOOOO < O0O0O000O0OO0OOOO:  # line:71
                            O0O000O0O0OO0OO0O = []  # line:72
                        O0O000O0O0OO0OO0O .append(O0000000OO000OO00)  # line:73
                        O00OO0000OOOOOOOO = O0O0O000O0OO0OOOO  # line:74
                else:  # line:75
                    if OOOOO00O0O0000OOO <= O0O0O000O0OO0OOOO:  # line:76
                        if O0O0O000O0OO0OOOO == 1000:  # line:77
                            return O0000000OO000OO00  # line:78
                        elif OOOOO00O0O0000OOO < O0O0O000O0OO0OOOO:  # line:79
                            O00000000O00OOO0O = []  # line:80
                        O00000000O00OOO0O .append(O0000000OO000OO00)  # line:81
                        OOOOO00O0O0000OOO = O0O0O000O0OO0OOOO  # line:82
            if len(O00000000O00OOO0O) > 0:  # line:83
                if random .random() < self .wall_prob:  # line:84
                    return random .choice(O00000000O00OOO0O)  # line:85
            return random .choice(O0O000O0O0OO0OO0O)  # line:87
