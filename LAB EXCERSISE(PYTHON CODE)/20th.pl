% Facts
fact(a).
fact(b).

% Rules
rule(c) :-
    fact(a),
    fact(b).

rule(d) :-
    rule(c).

rule(e) :-
    fact(b),
    rule(d).

% Forward Chaining
forward :-
    rule(X),
    write('Derived fact: '),
    write(X),
    nl,
    fail.

forward.