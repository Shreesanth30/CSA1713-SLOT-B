student(arun, cse, 21).
student(bala, it, 22).
student(chitra, cse, 20).
student(divya, ece, 21).

department(Name, Dept) :-
    student(Name, Dept, _).

above_age(Name, Age) :-
    student(Name, _, A),
    A > Age.