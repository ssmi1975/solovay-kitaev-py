from solovay_kitaev import solovay_kitaev as sk
import math

def test():
    t = sk.Uop(0, 1, 0, 0, 0, [])
    t = sk.Uop(math.cos(math.pi / 16), 0, 0, math.sin(math.pi / 16), 0, [], False);
    s = sk.generate_epsilon_network()
    print(t)
    for i in range(5):
        print(i)
        result = sk.solovay_kitaev(s, t, i)
        print(result.construction_str())


if __name__ == "__main__":
    test()
