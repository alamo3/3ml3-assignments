import matplotlib.pyplot as plt

def ooga_booga():

    file_text = open('word-list-7-letters.txt', 'r')

    words = file_text.readlines()

    for word in words:
        if word[0] == 'd' and word[-2] == 'd' and word[1] != 'e' and word[1] != 'a' and word.count('i') == 0:
            print(word, 'mom_'+word[0:-1]+'_me')


    # mom_dropped_me


def km_to_au(km):
    return km / 1.496e8


def gen_plot():
    list_planets = []
    list_planets.append((57.909e6, 0.206)) # Mercury
    list_planets.append((108.2e6, 0.007)) # Venus
    list_planets.append((149.6e6, 0.017)) # Earth
    list_planets.append((228e6, 0.094)) # Mars
    list_planets.append((778.5e6, 0.049)) # Jupiter
    list_planets.append((1432e6, 0.052)) # Saturn
    list_planets.append((2867e6, 0.047)) # Uranus
    list_planets.append((4515e6, 0.010)) # Neptune
    list_planets.append((5906.4e6, 0.244)) # Pluto

    plt.xscale(value='log')
    plt.xlabel('Orbital Semi-Major Axis (AU)')
    plt.ylabel('Orbital Eccentricity')
    plt.scatter([km_to_au(x[0]) for x in list_planets], [x[1] for x in list_planets])

    plt.show()


if __name__ == "__main__":
    gen_plot()