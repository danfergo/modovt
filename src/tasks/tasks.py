def sae(q1, q2):
    assert len(q1) == len(q2)
    return sum([abs(q1[i] - q2[i]) for i in range(len(q1))])


def reward(state):
    ws = [
        [0.1, 0.7],
        [0.3, 0.7],
        [0.2, 0.5],
    ]

    def is_outside_ws(xyz):
        return any([xyz[i] < ws[i][0] or xyz[i] > ws[i][1] for i in range(3)])

    def constraint_inside_ws(self, xyz):
        return [max(self.ws[i][0], min(xyz[i], self.ws[i][1])) for i in range(3)]

    g_delta = sae(state, [0.45, 0.48, 0.2])

    if g_delta < 0.2:
        return 200 - 100 * g_delta
    elif is_outside_ws(state):
        return -100
    else:
        return -1


grasp_rope = Task(
    reward=reward
)
