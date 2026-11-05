% Bird Flying Database

bird(sparrow).
bird(parrot).
bird(pigeon).
bird(ostrich).
bird(penguin).
bird(eagle).

can_fly(sparrow).
can_fly(parrot).
can_fly(pigeon).
can_fly(eagle).

cannot_fly(ostrich).
cannot_fly(penguin).

% Rule to check flying ability

check_bird(X) :-
    can_fly(X),
    write(X), write(' can fly.').

check_bird(X) :-
    cannot_fly(X),
    write(X), write(' cannot fly.').

check_bird(X) :-
    \+ bird(X),
    write('Bird not found in database.').