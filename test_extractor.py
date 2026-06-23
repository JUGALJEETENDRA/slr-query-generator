from contribution_extractor import extract_contribution

title = input("TITLE: ")
abstract = input("ABSTRACT: ")

result = extract_contribution(
    title=title,
    abstract=abstract
)

print(result)