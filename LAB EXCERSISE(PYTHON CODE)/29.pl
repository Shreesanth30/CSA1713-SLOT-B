% Symptoms
symptom(fever).
symptom(cough).
symptom(headache).
symptom(sneezing).
symptom(body_pain).

% Disease Rules

diagnosis(flu) :-
    has_symptom(fever),
    has_symptom(cough),
    has_symptom(body_pain).

diagnosis(common_cold) :-
    has_symptom(sneezing),
    has_symptom(cough).

diagnosis(malaria) :-
    has_symptom(fever),
    has_symptom(headache),
    has_symptom(body_pain).

% Patient Symptoms
has_symptom(fever).
has_symptom(cough).
has_symptom(body_pain).
has_symptom(headache).