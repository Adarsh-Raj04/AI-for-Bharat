import subprocess
import time
import sys

queries = [
"crohn disease treatment",
"ulcerative colitis therapy",
"irritable bowel syndrome treatment",
"gastroesophageal reflux disease therapy",
"peptic ulcer disease treatment",
"liver cirrhosis therapy",
"nonalcoholic fatty liver disease treatment",
"hepatocellular carcinoma therapy",
"celiac disease treatment",
"rheumatoid arthritis therapy",
"systemic lupus erythematosus treatment",
"psoriasis therapy",
"ankylosing spondylitis treatment",
"vasculitis therapy",
"sjogren syndrome treatment",
"autoimmune hepatitis therapy",
"chronic kidney disease treatment",
"acute kidney injury therapy",
"kidney transplantation outcomes",
"nephrotic syndrome treatment",
"kidney stones therapy",
"benign prostatic hyperplasia treatment",
"major depressive disorder therapy",
"bipolar disorder treatment",
"schizophrenia therapy",
"anxiety disorder treatment",
"post traumatic stress disorder therapy",
"attention deficit hyperactivity disorder treatment",
"autism spectrum disorder therapy",
"endometriosis treatment",
"preeclampsia management",
"gestational diabetes treatment",
"menopause hormone therapy",
"infertility treatment",
"cystic fibrosis therapy",
"muscular dystrophy treatment",
"sickle cell disease therapy",
"thalassemia treatment"
]

for i, query in enumerate(queries, 1):

    print(f"\nRunning query {i}/{len(queries)}: {query}")

    command = [
    sys.executable,
    "ingest_all.py",
    "--all",
    "--query",
    query,
    "--max-results",
    "100"
]

    subprocess.run(command)

    # small delay to avoid API throttling
    time.sleep(5)

print("\nAll ingestion jobs finished.")