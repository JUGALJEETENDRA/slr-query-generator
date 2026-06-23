from contribution_extractor import extract_contribution
from decision_engine import make_decision

rq = "Can large language models help automate systematic literature reviews?"

title = input("TITLE: ")
abstract = input("ABSTRACT: ")

x = extract_contribution(
    title=title,
    abstract=abstract
)

result = make_decision(
    research_question=rq,
    paper_topic=x["paper_topic"],
    paper_contribution=x["paper_contribution"],
    paper_task=x["paper_task"]
)

print("\nEXTRACTOR:")
print(x)

print("\nDECISION:")
print(result)