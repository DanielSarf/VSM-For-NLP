from helper_functions import generate_random_text

random_text = generate_random_text(500)

file_name = "random_spam.txt"

with open(file_name, 'w') as file:
    file.write(random_text)

print(f'Random spam text has been saved to {file_name}')