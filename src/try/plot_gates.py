import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = 'gates.npy'
    gates = np.load(filename)
    print(gates)
    plt.imsave('gates.jpeg', gates)


if __name__ == '__main__':
    main()






