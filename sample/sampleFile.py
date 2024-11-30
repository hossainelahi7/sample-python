# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
print("Hello world")

# age = 20
sentence = "this is a sentence."
word = "hug"
age1, age2, age3 = 17, 18, 19
ages = [age1, age2, age3]

for age in ages:
    print(age)

print(f"max {max(ages)}")
print(f"min {min(ages)}")

print(sentence)
print(sentence[1])
print(sentence[0:10])
name = "hossain"
age = 20
sentence = "%s is %d years old"
print(sentence % (name, age))

print(f"Hello, {name}")
print(f"All our age sum is {age1 + age2 + age3}")

# list vs dictionary

# dictionary

dictionary = {"asdsa", 1, "asd", 3, 4}
dictionary2 = {"Hossain": 22, "Muyeed": 20}
dictionary2["Hossain"] = 35
print(dictionary2)

# Triplet - not immutable
tup = ("hossain", "muyeed", "radiba")

number = 0
for number in range(10):
    if number == 5:
        pass  # pass here
    print('Number is ' + str(number))

print('Out of loop')

print(help("hello".upper()))

# sent = "print('hi')"
# eval(sent)
