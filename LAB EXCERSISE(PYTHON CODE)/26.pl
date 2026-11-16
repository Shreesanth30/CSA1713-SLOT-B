% Initial State
initial_state(state(at_door, on_floor, at_window, has_not)).

% Monkey Moves to Box
move(state(at_door, on_floor, at_window, has_not),
     state(at_window, on_floor, at_window, has_not)).

% Monkey Pushes Box
move(state(at_window, on_floor, at_window, has_not),
     state(at_middle, on_floor, at_middle, has_not)).

% Monkey Climbs Box
move(state(at_middle, on_floor, at_middle, has_not),
     state(at_middle, on_box, at_middle, has_not)).

% Monkey Grabs Banana
move(state(at_middle, on_box, at_middle, has_not),
     state(at_middle, on_box, at_middle, has)).

% Goal State
goal(state(_, _, _, has)).

% Solve Problem
solve(State, []) :-
    goal(State).

solve(State, [Action|Rest]) :-
    move(State, NextState),
    Action = move(State, NextState),
    solve(NextState, Rest).