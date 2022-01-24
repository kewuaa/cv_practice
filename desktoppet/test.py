import asyncio


class Test(object):
    def __init__(self):
        super(Test, self).__init__()
        self.state = False

    def __iter__(self):
        if not self.state:
            yield 1
        else:
            return 2


test = Test()


def s():
    a = yield from test
    print(a)


ss = s()
print(ss.send(None))
print(ss.send(None))
print(ss.send(None))
test.state = True
print(ss.send(None))
