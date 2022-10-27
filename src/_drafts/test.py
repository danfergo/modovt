from collections.abc import Mapping


class X(Mapping):

    def __init__(self):
        self.data = {
            'a': 11,
            'b': 22,
            'c': 33,
        }

    def __iter__(self):
        return iter(self.data)

    #
    # def __next__(self):
    #     if self.i > len(self.data.keys()) - 1:
    #         raise StopIteration
    #     el = list(self.data.keys())[self.i]
    #     # print(self.data.keys())
    #     # print(self.i)
    #     print('next', el)
    #     self.i += 1
    #     return el

    def __len__(self):
        print('len')
        return len(self.data.keys())

    def __getitem__(self, item):
        print('item', item)
        return self.data[item]


xx = X()


def fn(a, c, b, **e):
    print(a, b, c)
    print(e)

fn(**xx)

# a, b, c = xx
# print('-->', a, b, c)
#
# xyz = {'x': 1, 'y': 2, 'z': 3}
# *x, y, z = xyz
# print('-->', x)
