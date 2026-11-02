
% Facts
student(101, shreesanth).
student(102, rahul).
student(103, priya).

teacher(t1, ramesh).
teacher(t2, kumar).
teacher(t3, priya_teacher).

subject(csa0917, artificial_intelligence).
subject(csa0918, data_structures).
subject(csa0919, prolog).

% Relationships
teaches(t1, csa0917).
teaches(t2, csa0918).
teaches(t3, csa0919).

enrolled(101, csa0917).
enrolled(102, csa0918).
enrolled(103, csa0919).

% Rules
student_teacher(Student, Teacher) :-
    enrolled(Student, Subject),
    teaches(Teacher, Subject).

student_subject(Student, Subject) :-
    enrolled(Student, Subject).

teacher_subject(Teacher, Subject) :-
    teaches(Teacher, Subject).

