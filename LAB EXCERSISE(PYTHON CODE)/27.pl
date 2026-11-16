% Fruit and Color Facts

fruit(apple).
fruit(banana).
fruit(orange).
fruit(grapes).
fruit(mango).

color(apple, red).
color(banana, yellow).
color(orange, orange).
color(grapes, green).
color(mango, yellow).

% Rule
fruit_color(Fruit, Color) :-
    fruit(Fruit),
    color(Fruit, Color).