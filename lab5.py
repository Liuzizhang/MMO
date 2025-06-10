import spacy

# Загружаем модель spacy для английского языка
nlp = spacy.load("en_core_web_sm")

# Пример произвольного предложения
text = "Apple is looking at buying U.K. startup for $1 billion."

# Пропускаем текст через пайплайн spacy
doc = nlp(text)

# Токенизация
print("Tokens:")
for token in doc:
    print(f"{token.text}", end=" | ")
print("\n")

# Частеречная разметка
print("POS tagging:")
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")
print("\n")

# Лемматизация
print("Lemmas:")
for token in doc:
    print(f"{token.text} -> {token.lemma_}")
print("\n")

# Именованные сущности
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
print("\n")

# Синтаксический разбор
print("Dependency Parsing:")
for token in doc:
    print(f"{token.text} --> {token.dep_} --> {token.head.text}")
