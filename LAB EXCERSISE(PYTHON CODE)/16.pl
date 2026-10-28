% Program to find the sum of integers from 1 to N

sum(0, 0).

sum(N, S) :-
N > 0,
N1 is N - 1,
sum(N1, S1),
S is N + S1.

start :-
write('Enter a number: '),
read(N),
sum(N, S),
write('Sum = '),
write(S),
nl.
