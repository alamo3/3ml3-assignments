

file_text = open('word-list-7-letters.txt', 'r')

words = file_text.readlines()

for word in words:
    if word[0] == 'd' and word[-2] == 'd' and word[1] != 'e' and word[1] != 'a' and word.count('i') == 0:
        print(word, 'mom_'+word[0:-1]+'_me')


# mom_dropped_me