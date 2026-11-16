% Family Tree Facts

male(john).
male(peter).
male(james).
male(robert).

female(mary).
female(lisa).
female(susan).
female(anna).

% Parent Relationships

parent(john, peter).
parent(mary, peter).

parent(john, lisa).
parent(mary, lisa).

parent(peter, james).
parent(susan, james).

parent(peter, anna).
parent(susan, anna).

% Rules

father(X,Y) :-
    parent(X,Y),
    male(X).

mother(X,Y) :-
    parent(X,Y),
    female(X).

child(X,Y) :-
    parent(Y,X).

sibling(X,Y) :-
    parent(Z,X),
    parent(Z,Y),
    X \= Y.

brother(X,Y) :-
    sibling(X,Y),
    male(X).

sister(X,Y) :-
    sibling(X,Y),
    female(X).