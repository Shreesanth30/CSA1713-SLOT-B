% Graph Connections
edge(a,b).
edge(a,c).
edge(b,d).
edge(b,e).
edge(c,f).
edge(c,g).
edge(d,h).
edge(e,h).
edge(f,h).
edge(g,h).

% Heuristic Values
heuristic(a,7).
heuristic(b,6).
heuristic(c,4).
heuristic(d,5).
heuristic(e,3).
heuristic(f,2).
heuristic(g,6).
heuristic(h,0).

% Best First Search
best_first(Start, Goal, Path) :-
    search([[Start]], Goal, ReversePath),
    reverse(ReversePath, Path).

search([[Goal|Rest]|_], Goal, [Goal|Rest]).

search([CurrentPath|OtherPaths], Goal, FinalPath) :-
    CurrentPath = [Current|_],
    findall(
        [Next|CurrentPath],
        (edge(Current,Next),
         \+ member(Next,CurrentPath)),
        NewPaths
    ),
    append(OtherPaths, NewPaths, AllPaths),
    sort_paths(AllPaths, SortedPaths),
    search(SortedPaths, Goal, FinalPath).

sort_paths(Paths, Sorted) :-
    predsort(compare_paths, Paths, Sorted).

compare_paths(Order, [A|_], [B|_]) :-
    heuristic(A, HA),
    heuristic(B, HB),
    compare(Order, HA, HB).