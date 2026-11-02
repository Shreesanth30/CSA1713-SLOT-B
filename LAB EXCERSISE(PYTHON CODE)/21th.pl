% Facts
parent(john, mary).
parent(mary, alice).
parent(john, david).

% Rules
grandparent(X, Y) :-
    parent(X, Z),
    parent(Z, Y).

ancestor(X, Y) :-
    parent(X, Y).

ancestor(X, Y) :-
    parent(X, Z),
    ancestor(Z, Y).

% Backward Chaining
backward(Goal) :-
    call(Goal).