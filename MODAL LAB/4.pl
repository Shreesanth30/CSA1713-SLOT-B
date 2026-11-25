connected(chennai, bangalore).
connected(chennai, hyderabad).
connected(bangalore, mumbai).
connected(hyderabad, delhi).
connected(mumbai, delhi).
connected(delhi, kolkata).

path(X, Y, [X, Y]) :-
    connected(X, Y).

path(X, Y, [X|P]) :-
    connected(X, Z),
    path(Z, Y, P).