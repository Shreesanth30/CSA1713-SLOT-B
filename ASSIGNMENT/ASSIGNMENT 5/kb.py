"""College knowledge base (synthetic stand-in for the 5 PDFs in the REVIEW_2 deck) + retriever.
Retriever = TF-IDF + cosine over an inner-product index (stand-in for MiniLM + FAISS IndexFlatIP, offline)."""
import random, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DEPTS = {"BIA": "Biotechnology", "CSC": "Computer Science", "ITA": "Information Technology",
         "ECE": "Electronics and Communication", "MEC": "Mechanical Engineering", "MAT": "Mathematics"}
NAMES = {"BIA": ["Cell Biology", "Genetics", "Biochemistry", "Microbiology", "Bioinformatics", "Immunology", "Molecular Biology"],
         "CSC": ["Python Programming", "Data Structures", "Operating Systems", "Computer Networks", "Machine Learning", "Compiler Design", "Database Systems"],
         "ITA": ["Web Technologies", "Cloud Computing", "Cyber Security", "Mobile Application Development", "Software Testing", "Data Analytics", "Internet of Things"],
         "ECE": ["Digital Electronics", "Signals and Systems", "Embedded Systems", "VLSI Design", "Control Systems", "Antenna Theory", "Analog Circuits"],
         "MEC": ["Thermodynamics", "Fluid Mechanics", "Machine Design", "Manufacturing Processes", "Heat Transfer", "Robotics", "Engineering Mechanics"],
         "MAT": ["Linear Algebra", "Probability and Statistics", "Discrete Mathematics", "Numerical Methods", "Calculus", "Graph Theory", "Optimization Techniques"]}

def build_corpus(seed=42):
    rnd = random.Random(seed)
    chunks, qa, courses = [], [], []
    def add(doc, page, text):
        chunks.append({"id": len(chunks), "doc": doc, "page": page, "text": text}); return len(chunks) - 1
    n = 0
    for p, d in DEPTS.items():
        for i, nm in enumerate(NAMES[p]):
            n += 1
            code = f"{p}0{2 if i % 2 else 1}{i+1:02d}"
            if nm == "Cell Biology": code = "BIA0201"
            if nm == "Python Programming": code = "CSC0202"
            strength = 64 if code == "BIA0201" else rnd.randint(30, 120)
            sem = rnd.randint(1, 8); cr = rnd.choice([2, 3, 3, 4])
            courses.append(dict(code=code, name=nm, dept=d, sem=sem, credits=cr, strength=strength))
    for c in courses:
        a = add("Academic Information PDF", 1 + c["sem"] // 3, f"Course {c['code']} {c['name']} has a student strength of {c['strength']} students.")
        b = add("Course Catalogue PDF", 1 + c["sem"] // 3, f"{c['name']} ({c['code']}) is offered by the Department of {c['dept']} in Semester {c['sem']} and carries {c['credits']} credits.")
        qa += [dict(q=f"What is the strength of {c['code']} {c['name']}?", a=f"{c['strength']} students", src=a, kind="strength"),
               dict(q=f"How many students are enrolled in {c['name']}?", a=f"{c['strength']} students", src=a, kind="strength"),
               dict(q=f"What is the course code of {c['name']}?", a=c["code"], src=b, kind="code"),
               dict(q=f"Which department offers {c['name']}?", a=c["dept"], src=b, kind="dept"),
               dict(q=f"In which semester is {c['name']} offered?", a=f"Semester {c['sem']}", src=b, kind="semester"),
               dict(q=f"How many credits does {c['code']} carry?", a=f"{c['credits']} credits", src=b, kind="credits")]
    events = ["CIA-1 examination", "CIA-2 examination", "End semester examination", "Practical examination"]
    for s in range(1, 5):
        for k, ev in enumerate(events):
            d1 = 3 + 5 * k + s; m = ["Aug", "Sep", "Oct", "Nov"][k]
            t = f"The {ev} for Semester {s} is scheduled from {d1} {m} 2026 to {d1+4} {m} 2026."
            i = add("Academic Calendar PDF", 1 + s // 2, t)
            qa.append(dict(q=f"When is the {ev} for Semester {s}?", a=f"{d1} {m} 2026", src=i, kind="calendar"))
    rules = [("minimum attendance", "Students must maintain a minimum attendance of 75% in every course to be eligible to write the end semester examination.", "75%",
              ["What is the minimum attendance requirement?", "How much attendance is needed to write the exam?", "What percentage of attendance is mandatory?"]),
             ("condonation", "Attendance between 65% and 74% may be condoned on medical grounds with a valid certificate and payment of the condonation fee.", "65%",
              ["What is the attendance condonation range?", "Can low attendance be condoned for medical reasons?"]),
             ("hall ticket", "The hall ticket must be carried along with the college identity card to every examination.", "hall ticket",
              ["What must students carry to the exam hall?", "Is the hall ticket compulsory for examinations?"]),
             ("pass mark", "The minimum pass mark in the end semester examination is 50 percent of the maximum marks.", "50 percent",
              ["What is the minimum pass mark?", "What score is needed to pass the end semester examination?"]),
             ("malpractice", "Any student found involved in malpractice during an examination will be debarred from the examination and reported to the disciplinary committee.", "debarred",
              ["What is the penalty for malpractice in exams?", "What happens if a student cheats in an examination?"]),
             ("revaluation", "Applications for revaluation must be submitted within 7 days of the publication of results with a fee of 500 rupees per paper.", "500 rupees",
              ["What is the fee for revaluation?", "How much do I pay to apply for revaluation of a paper?"]),
             ("reappear", "Students with arrears may appear for supplementary examinations conducted in the following semester.", "supplementary",
              ["How can students clear arrears?", "When can students with arrear papers write the exam?"]),
             ("CIA weightage", "Continuous internal assessment carries 40 percent weightage and the end semester examination carries 60 percent weightage.", "40 percent",
              ["What is the weightage of continuous internal assessment?", "How many marks does internal assessment carry?"])]
    for key, t, a, qs in rules:
        i = add("Examination Guidelines PDF", 1 + len(chunks) % 3, t)
        for q in qs: qa.append(dict(q=q, a=a, src=i, kind="rule"))
    # redundant fact in another document (needed for the 'source deleted' scenario)
    add("Student Services FAQ PDF", 1, "Please note that every student needs at least 75% attendance to be allowed into the end semester examination.")
    faq = [("library", "The central library is open from 8 AM to 8 PM on working days and 9 AM to 1 PM on Saturdays.", "8 AM to 8 PM", ["What are the library timings?", "When does the library open and close?"]),
           ("placement", "The placement cell conducts aptitude training and campus recruitment drives from the sixth semester onwards.", "sixth semester", ["From which semester does the placement training start?", "When do campus recruitment drives begin?"]),
           ("it support", "For wifi password reset or email access issues, raise a ticket on the IT helpdesk portal or visit the IT support room in Block A.", "IT helpdesk portal", ["How do I reset my wifi password?", "Where can I get IT support for email problems?"]),
           ("scholarship", "Merit scholarship applications are accepted in July and require the previous semester mark statement.", "July", ["When are scholarship applications accepted?", "What document is needed for the merit scholarship?"]),
           ("hostel", "Hostel allotment requests are processed by the warden office within 5 working days of application.", "5 working days", ["How long does hostel allotment take?", "Who processes hostel requests?"]),
           ("bonafide", "Bonafide certificates are issued by the academic office within 2 working days on request.", "2 working days", ["How long does it take to get a bonafide certificate?", "Which office issues bonafide certificates?"]),
           ("transport", "College buses run on 14 routes and the transport fee is collected at the beginning of every semester.", "14 routes", ["How many bus routes does the college operate?", "When is the transport fee collected?"]),
           ("counselling", "Student counselling services are available free of charge at the wellness centre between 10 AM and 4 PM.", "wellness centre", ["Where can students get counselling?", "Is student counselling free?"])]
    for key, t, a, qs in faq:
        i = add("Student Services FAQ PDF", 1 + len(chunks) % 2, t)
        for q in qs: qa.append(dict(q=q, a=a, src=i, kind="faq"))
    oos = ["Who won the cricket world cup?", "What is the capital of France?", "Tell me a joke about programming.", "How do I cook biryani?",
           "What is the weather in Chennai today?", "Who is the prime minister of India?", "Explain quantum entanglement.", "What is the price of gold today?",
           "Recommend a good movie for the weekend.", "How do I train for a marathon?", "Who wrote Hamlet?", "What is the stock price of Tesla?",
           "Translate good morning into Japanese.", "How tall is Mount Everest?", "What is the best smartphone to buy?", "Write a poem about the sea.",
           "How many moons does Jupiter have?", "Who is the richest person in the world?", "Plan a trip to Paris for me.", "What is bitcoin trading at?"]
    return chunks, qa, courses, oos

class VectorIndex:
    """TF-IDF vectors L2-normalised -> inner product == cosine (what FAISS IndexFlatIP does on normalised MiniLM vectors)."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.M = self.vec.fit_transform([c["text"] for c in chunks])
    def search(self, query, k=3, exclude_docs=(), exclude_ids=()):
        q = self.vec.transform([query]); s = (self.M @ q.T).toarray().ravel()
        for c in self.chunks:
            if c["doc"] in exclude_docs or c["id"] in exclude_ids: s[c["id"]] = -1
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

def extractive_answer(query, text):
    """Deterministic stand-in for Llama 3.2: return the best-matching sentence of the context."""
    sents = re.split(r"(?<=[.!?])\s+", text); qw = set(re.findall(r"\w+", query.lower()))
    return max(sents, key=lambda s: len(qw & set(re.findall(r"\w+", s.lower()))))
